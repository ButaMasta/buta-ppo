// buta-ppo/src/env/rocketsim_env.cpp
#include "rocketsim_env.hpp"
#include "buta_ppo/ffi.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iterator>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace buta_ppo::reward;

namespace buta_ppo::env {

RocketSimEnv::RocketSimEnv(
    const std::vector<ffi::Team>& match_layout, 
    const std::vector<state::StateSetterDistribution>& setters, 
    const std::vector<reward::RewardEntry>& rewards,
    const std::vector<metric::MetricEntry>& metrics,
    int ticks_per_step, const EpisodeLimits& limits, size_t max_players_per_team,
    uint32_t seed, 
    bool render,
    std::optional<ffi::Team> opponent_team,
    const std::vector<external::ExternalSide>& external_sides
) : opponent_team_(opponent_team),
    setters_(setters),
    rng_(seed),
    ticks_per_step_(ticks_per_step),
    limits_(limits),
    obs_builder_(max_players_per_team, ticks_per_step, seed), 
    render_(render) {

    // The FFI arena state has a fixed number of car slots, shared by both teams.
    constexpr size_t max_cars = std::extent_v<decltype(ffi::ArenaState::cars)>;
    if (max_players_per_team > max_cars / 2) {
        throw std::invalid_argument("RocketSimEnv: max_players_per_team can be at most " + std::to_string(max_cars / 2) + ".");
    }

    if (setters_.empty()) {
        throw std::invalid_argument("RocketSimEnv: setter_distributions must contain at least one state setter.");
    }
    double setter_weight_sum = 0.0;
    for (const auto& dist : setters_) {
        if (dist.weight < 0.0f) {
            throw std::invalid_argument("RocketSimEnv: state setter weights must be non-negative.");
        }
        setter_weight_sum += dist.weight;
    }
    if (setter_weight_sum <= 0.0) {
        throw std::invalid_argument("RocketSimEnv: sum of state setter weights must be positive.");
    }

    arena_ = render ? ffi::create_arena_vis(0) : ffi::create_arena(0);
    if (!arena_) {
        throw std::runtime_error(render ? "Could not start the RocketSim viewer. Install it with `cargo binstall "
                                          "rocketsim-viewer@0.3.3` or `cargo install --locked rocketsim_viewer --version "
                                          "0.3.3` and put rocketsim-viewer on PATH, or set ROCKETSIM_VIEWER to its path "
                                          "(see docs/visualizer.md)."
                                        : "Could not create a RocketSim arena.");
    }
    single_obs_size_ = obs_builder_.get_obs_size();
    action_space_size_ = action_parser_.get_action_space_size();

    // Fill teams.
    const size_t blue_count = static_cast<size_t>(std::count(match_layout.begin(), match_layout.end(), ffi::Team::Blue));
    if (blue_count > max_players_per_team || match_layout.size() - blue_count > max_players_per_team) {
        throw std::invalid_argument("RocketSimEnv: match layout has a team larger than max_players_per_team.");
    }

    for (const ffi::Team team : match_layout) {
        add_agent(team);
    }

    // Hand the external bots' teams to their controllers.
    for (const external::ExternalSide& side : external_sides) {
        if (!side.bot) {
            throw std::invalid_argument("RocketSimEnv: an external side has no bot.");
        }

        ExternalGroup group;
        group.bot = side.bot;
        for (size_t i = 0; i < agents_.size(); i++) {
            if (agents_[i].team != side.team) continue;
            if (controllers_[i]) {
                throw std::invalid_argument("RocketSimEnv: two external bots were given the same team.");
            }
            controllers_[i] = side.bot->make_controller(agents_[i].car_id);
            group.agents.push_back(i);
        }
        group.deciding.reserve(group.agents.size());
        group.obs.resize(group.agents.size() * side.bot->obs_size(agents_.size()));
        group.decisions.resize(group.agents.size() * side.bot->decision_size());
        external_groups_.push_back(std::move(group));
    }
    if (!external_groups_.empty()) {
        size_lockstep_buffers();
    }

    // Initial shuffle of indices.
    obs_builder_.team_indices_rand(agents_);

    // Construct state setter dist.
    std::vector<double> weights;
    weights.reserve(setters_.size());
    for (const auto& dist : setters_) {
        weights.push_back(dist.weight);
    }
    setter_selector_ = std::discrete_distribution<size_t>(weights.begin(), weights.end());

    // Setup Rewards.
    reward_manager_.set_rewards(rewards);
    // Setup Metrics.
    metric_manager_.set_metrics(metrics);
}

uint32_t RocketSimEnv::add_agent(ffi::Team team) {
    const uint32_t car_id = ffi::add_car(arena_, team);
    agents_.push_back({car_id, team});

    // Resize buffers for the new agent.
    obs_buffer_.resize(agents_.size() * single_obs_size_, 0.0f);
    action_mask_buffer_.resize(agents_.size() * action_space_size_, 0.0f);
    reward_buffer_.resize(agents_.size(), 0.0f);
    agent_x_inverted_.resize(agents_.size(), false);
    controllers_.resize(agents_.size());

    return car_id;
}

ResetResult RocketSimEnv::reset() {
    reset_episode_state();
    if (!external_groups_.empty()) {
        start_external_episode();
    }
    build_all_obs();

    return { obs_buffer_, action_mask_buffer_ };
}

StepResult RocketSimEnv::step(const int* actions) {
    apply_actions(actions);

    std::fill(reward_buffer_.begin(), reward_buffer_.end(), 0.0f);
    reward_manager_.begin_step();
    StepProgress progress{.ticks_elapsed = ticks_per_step_};

    for (int i = 0; i < ticks_per_step_; i++) {
        if (!external_groups_.empty()) {
            run_external_packet(true);
        }
        if (advance_tick(i, progress)) break;
    }

    return finish_step(progress);
}

void RocketSimEnv::reset_episode_state() {
    const size_t chosen_idx = setter_selector_(rng_);
    setters_[chosen_idx].setter->apply(arena_, arena_state_, agents_, rng_);
    // The setter teleported the cars and the ball. Once all are placed, the old contacts are dropped and the state is read
    // back with each car's grounding at its new pose, so the first obs sees it.
    ffi::settle_contacts(arena_, arena_state_);
    ticks_since_last_touch_ = 0;
    match_ticks_ = 0;
    reward_manager_.reset(agents_, arena_state_);
    metric_manager_.reset(arena_state_);
    obs_builder_.reset();
    obs_builder_.team_indices_rand(agents_);
}

void RocketSimEnv::apply_actions(const int* actions) {
    for (size_t i = 0; i < agents_.size(); i++) {
        // External bots set their controls every tick.
        if (controllers_[i]) continue;

        ffi::CarControls controls;
        if (heavy_workload_) [[unlikely]] {
            controls = chaser_controls(arena_state_.cars[agents_[i].car_id]);
        } else {
            controls = decode_action(actions[i]);

            // Invert necessary controls if the state was X-mirrored.
            if (agent_x_inverted_[i]) {
                controls.steer  *= -1.0f;
                controls.yaw    *= -1.0f;
                controls.roll   *= -1.0f;
            }
        }

        ffi::set_car_controls(arena_, agents_[i].car_id, controls);
    }
}

bool RocketSimEnv::advance_tick(int tick, StepProgress& progress) {
    ffi::step_arena(arena_);
    ffi::get_arena_state(arena_, arena_state_);

    if (render_) [[unlikely]] {
        std::this_thread::sleep_for(std::chrono::microseconds(8333));
    }

    // Accumulate rewards and metrics. Rewards also update their previous state.
    reward_manager_.accumulate_tick(agents_, arena_state_);
    for (size_t a = 0; a < agents_.size(); a++) {
        // opponent_team_ is an optional only set by pfsp envs. If it is not set then this evaluates to true.
        const bool track_telemetry = agents_[a].team != opponent_team_;
        metric_manager_.calculate_metrics(agents_[a], arena_state_, track_telemetry);
    }
    metric_manager_.update_previous_state(arena_state_);

    // Reset touch counter if any car hit the ball or increment it.
    if (std::any_of(std::begin(arena_state_.events.car_hit_ball), std::end(arena_state_.events.car_hit_ball), [](bool v) { return v; })) {
        ticks_since_last_touch_ = 0;
        ++touch_ticks_;
        kickoff_pause_ = false;
    } else {
        ++ticks_since_last_touch_;
    }
    // Increment match ticks.
    ++match_ticks_;

    // Terminal state: Goal.
    if (arena_state_.events.is_ball_scored) {
        // The ball is past orange's goal line (positive Y) when blue scores.
        progress.blue_scored = arena_state_.ball.phys.pos[1] > 0;
        progress.ticks_elapsed = tick + 1;
        progress.terminated = true;
        return true;
    }

    // Tick count related truncation states. A limit of 0 is off.
    const bool no_touch = limits_.no_touch_ticks > 0 && ticks_since_last_touch_ >= limits_.no_touch_ticks;
    const bool time_up = limits_.time_limit_ticks > 0 && match_ticks_ >= limits_.time_limit_ticks;
    if (no_touch || time_up) {
        progress.no_touch_truncated = no_touch;
        progress.ticks_elapsed = tick + 1;
        progress.truncated = true;
        return true;
    }
    return false;
}

void RocketSimEnv::build_all_obs() {
    for (size_t i = 0; i < agents_.size(); i++) {
        float* agent_obs_ptr = obs_buffer_.data() + (i * single_obs_size_);
        agent_x_inverted_[i] = obs_builder_.build_obs(arena_state_, agents_, static_cast<uint32_t>(i), agent_obs_ptr);

        const ffi::CarState& car_state = arena_state_.cars[agents_[i].car_id];
        float* agent_action_mask_ptr = action_mask_buffer_.data() + (i * action_space_size_);
        action_parser_.get_action_mask(car_state, agent_action_mask_ptr);
    }
}

StepResult RocketSimEnv::finish_step(const StepProgress& progress) {
    // Apply the reward options to the step's rewards. The opponent team is left out of the telemetry.
    reward_manager_.end_step(reward_buffer_, opponent_team_);

    // Build the final observations.
    build_all_obs();

    return { obs_buffer_, action_mask_buffer_, reward_buffer_, progress.terminated, progress.blue_scored,
             progress.truncated, progress.no_touch_truncated, progress.ticks_elapsed };
}

int RocketSimEnv::lockstep_reset() {
    if (external_groups_.empty()) {
        throw std::logic_error("RocketSimEnv: lockstep play needs an external bot.");
    }
    reset_episode_state();

    // Fresh controllers, which join the match anew like bots starting up for this game.
    for (ExternalGroup& group : external_groups_) {
        for (const size_t agent : group.agents) {
            controllers_[agent] = group.bot->make_controller(agents_[agent].car_id);
        }
        group.pending = 0;
    }
    external_started_ = false;
    pending_.clear();
    size_lockstep_buffers();

    begin_frozen_packets();
    request_external_decisions();
    return frozen_packets_left_;
}

std::optional<ResetResult> RocketSimEnv::lockstep_frozen(int packets) {
    set_pending_decisions();
    for (int k = 0; k < packets; k++) {
        tick_external(false);
        end_frozen_packet();
        if (frozen_packets_left_ == 0) {
            build_all_obs();
            // The first tick's packet.
            request_external_decisions();
            return ResetResult{ obs_buffer_, action_mask_buffer_ };
        }
        request_external_decisions();
    }
    return std::nullopt;
}

void RocketSimEnv::lockstep_begin_step(const int* actions) {
    apply_actions(actions);
    std::fill(reward_buffer_.begin(), reward_buffer_.end(), 0.0f);
    reward_manager_.begin_step();
    lockstep_progress_ = {.ticks_elapsed = ticks_per_step_};
    lockstep_tick_ = 0;
}

std::optional<StepResult> RocketSimEnv::lockstep_ticks(int ticks) {
    set_pending_decisions();
    for (int k = 0; k < ticks; k++) {
        tick_external(true);
        const bool ended = advance_tick(lockstep_tick_, lockstep_progress_);
        lockstep_tick_++;
        if (ended || lockstep_tick_ == ticks_per_step_) {
            std::optional<StepResult> result(finish_step(lockstep_progress_));
            if (!ended) {
                // The next step's first packet.
                request_external_decisions();
            }
            return result;
        }
        request_external_decisions();
    }
    return std::nullopt;
}

void RocketSimEnv::size_lockstep_buffers() {
    int delay = INT32_MAX;
    for (const ExternalGroup& group : external_groups_) {
        for (const size_t agent : group.agents) delay = std::min(delay, controllers_[agent]->decision_delay());
    }
    lockstep_window_ = std::max(1, delay);
    // A car asks for at most one decision per `decision_interval` packets of a window.
    for (ExternalGroup& group : external_groups_) {
        size_t per_car = 0;
        for (const size_t agent : group.agents) {
            const int interval = std::max(1, controllers_[agent]->decision_interval());
            per_car = std::max(per_car, static_cast<size_t>((lockstep_window_ + interval - 1) / interval));
        }
        group.max_pending = group.agents.size() * per_car;
        group.obs.resize(std::max(group.obs.size(), group.max_pending * group.bot->obs_size(agents_.size())));
        group.decisions.resize(std::max(group.decisions.size(), group.max_pending * group.bot->decision_size()));
    }
}

void RocketSimEnv::request_external_decisions() {
    const external::GameInfo info{ kickoff_pause_ };
    for (size_t a = 0; a < agents_.size(); a++) {
        if (external_joined(a)) controllers_[a]->observe(arena_state_, info);
    }

    for (size_t g = 0; g < external_groups_.size(); g++) {
        ExternalGroup& group = external_groups_[g];
        const size_t obs_size = group.bot->obs_size(agents_.size());
        for (const size_t agent : group.agents) {
            if (!external_joined(agent) || !controllers_[agent]->wants_decision()) continue;
            if (group.pending == group.max_pending) {
                throw std::logic_error("RocketSimEnv: an external bot asked for more decisions than its decision_interval allows.");
            }
            controllers_[agent]->build_obs(arena_state_, group.obs.data() + group.pending * obs_size);
            controllers_[agent]->request_decision();
            pending_.push_back({g, agent, group.pending});
            group.pending++;
        }
    }
}

void RocketSimEnv::set_pending_decisions() {
    for (const PendingDecision& p : pending_) {
        const ExternalGroup& group = external_groups_[p.group];
        controllers_[p.agent]->set_decision(group.decisions.data() + p.slot * group.bot->decision_size(), rng_);
    }
    pending_.clear();
    for (ExternalGroup& group : external_groups_) group.pending = 0;
}

void RocketSimEnv::tick_external(bool apply_controls) {
    for (size_t a = 0; a < agents_.size(); a++) {
        if (!external_joined(a)) continue;
        const ffi::CarControls controls = controllers_[a]->tick(arena_state_);
        if (apply_controls) {
            ffi::set_car_controls(arena_, agents_[a].car_id, controls);
        }
    }
}

RocketSimEnv::ExternalBatch RocketSimEnv::external_batch(size_t group) {
    ExternalGroup& g = external_groups_.at(group);
    return { g.bot.get(), agents_.size(), g.pending, g.obs.data(), g.decisions.data() };
}

void RocketSimEnv::set_heavy_workload(bool enabled) {
    heavy_workload_ = enabled;
    // A 1v1 bot that dribbles repredicts on about 1 in 6 steps. At least every 3rd step makes up for the scripted cars'
    // contacts being shorter and cheaper than a dribble's, bringing the env cost per step to about the dribbler's.
    obs_builder_.set_min_repred_interval(enabled ? 3 : 0);
}

ffi::CarControls RocketSimEnv::chaser_controls(const ffi::CarState& car) const {
    const float* ball = arena_state_.ball.phys.pos;
    const float to_ball[3] = { ball[0] - car.phys.pos[0], ball[1] - car.phys.pos[1], ball[2] - car.phys.pos[2] };
    const float* forward = car.phys.rot_mat[0];
    const float* right = car.phys.rot_mat[1];
    const float ahead = to_ball[0] * forward[0] + to_ball[1] * forward[1] + to_ball[2] * forward[2];
    const float side = to_ball[0] * right[0] + to_ball[1] * right[1] + to_ball[2] * right[2];
    const float angle = std::atan2(side, ahead);
    const float dist = std::sqrt(to_ball[0] * to_ball[0] + to_ball[1] * to_ball[1] + to_ball[2] * to_ball[2]);

    // Drive at the ball, boost when facing it, powerslide when it's behind, and jump at balls in the air nearby.
    ffi::CarControls controls{};
    controls.throttle = 1.0f;
    controls.steer = std::clamp(angle * 2.5f, -1.0f, 1.0f);
    controls.boost = std::abs(angle) < 0.3f && car.boost > 0.0f;
    controls.handbrake = std::abs(angle) > 1.6f;
    controls.jump = car.is_on_ground && dist < 300.0f && ball[2] > 180.0f;
    return controls;
}

void RocketSimEnv::set_external_bot(size_t group, std::shared_ptr<external::ExternalBot> bot) {
    ExternalGroup& g = external_groups_.at(group);
    if (!bot || bot->obs_size(agents_.size()) != g.bot->obs_size(agents_.size()) || bot->decision_size() != g.bot->decision_size()) {
        throw std::invalid_argument("RocketSimEnv::set_external_bot needs a bot with the same obs and decision sizes.");
    }
    g.bot = std::move(bot);
}

void RocketSimEnv::start_external_episode() {
    begin_frozen_packets();
    while (frozen_packets_left_ > 0) {
        run_external_packet(false);
        end_frozen_packet();
    }
}

void RocketSimEnv::begin_frozen_packets() {
    // A kickoff state: the ball resting at center field.
    const ffi::PhysState& ball = arena_state_.ball.phys;
    frozen_kickoff_ = ball.pos[0] == 0.0f && ball.pos[1] == 0.0f
                   && ball.vel[0] == 0.0f && ball.vel[1] == 0.0f && ball.vel[2] == 0.0f;

    // Frozen packets aren't paced in render: nothing moves, and the visualizer shows nothing until play starts.
    kickoff_pause_ = false;

    // Each bot joins its first match at its own arbitrary tick, like a bot process starting up, which sets the phase of
    // any tick cycle it keeps. So bots in the same match don't decide on the same ticks.
    join_packets_left_ = 0;
    if (!external_started_) {
        std::uniform_int_distribution<int> join_tick(0, JOIN_WINDOW_TICKS - 1);
        join_ticks_.assign(agents_.size(), 0);
        for (size_t a = 0; a < agents_.size(); a++) {
            if (controllers_[a]) join_ticks_[a] = join_tick(rng_);
        }
        join_packets_left_ = JOIN_WINDOW_TICKS;
    }

    // In game, bots keep getting packets through the countdown before a kickoff, with every car frozen. It's a whole
    // number of 8-tick cycles, so the bots' phases carry through it.
    frozen_packets_left_ = join_packets_left_ + (frozen_kickoff_ ? KICKOFF_COUNTDOWN_TICKS : 0);
    if (frozen_packets_left_ == 0) {
        kickoff_pause_ = frozen_kickoff_;
    }
}

void RocketSimEnv::end_frozen_packet() {
    if (join_packets_left_ > 0) {
        for (int& ticks : join_ticks_) {
            if (ticks > 0) ticks--;
        }
        if (--join_packets_left_ == 0) {
            join_ticks_.clear();
            external_started_ = true;
        }
    }
    // "Go": the kickoff pause lasts until the first touch.
    if (--frozen_packets_left_ == 0) {
        kickoff_pause_ = frozen_kickoff_;
    }
}

void RocketSimEnv::run_external_packet(bool apply_controls) {
    const external::GameInfo info{ kickoff_pause_ };
    for (size_t a = 0; a < agents_.size(); a++) {
        if (external_joined(a)) controllers_[a]->observe(arena_state_, info);
    }
    decide_external();
    tick_external(apply_controls);
}

void RocketSimEnv::decide_external() {
    for (ExternalGroup& group : external_groups_) {
        group.deciding.clear();
        for (const size_t agent : group.agents) {
            if (external_joined(agent) && controllers_[agent]->wants_decision()) group.deciding.push_back(agent);
        }
        if (group.deciding.empty()) continue;

        const size_t obs_size = group.bot->obs_size(agents_.size());
        const size_t decision_size = group.bot->decision_size();
        for (size_t k = 0; k < group.deciding.size(); k++) {
            controllers_[group.deciding[k]]->build_obs(arena_state_, group.obs.data() + k * obs_size);
        }
        group.bot->infer(group.obs.data(), group.deciding.size(), agents_.size(), group.decisions.data());
        for (size_t k = 0; k < group.deciding.size(); k++) {
            controllers_[group.deciding[k]]->set_decision(group.decisions.data() + k * decision_size, rng_);
        }
    }
}

ffi::CarControls RocketSimEnv::decode_action(size_t action_idx) {
    return action_parser_.get_action(action_idx);
}

}; // namespace buta_ppo::env