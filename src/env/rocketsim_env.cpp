// buta-ppo/src/env/rocketsim_env.cpp
#include "rocketsim_env.hpp"
#include "buta_ppo/ffi.h"

#include <algorithm>
#include <thread>
#include <chrono>

using namespace buta_ppo::reward;

namespace buta_ppo::env {

RocketSimEnv::RocketSimEnv(
    const std::vector<ffi::Team>& match_layout, 
    const std::vector<state::StateSetterDistribution>& setters, 
    const std::vector<reward::RewardEntry>& rewards,
    int ticks_per_step, size_t max_players_per_team, 
    uint32_t seed, 
    bool render
) : setters_(setters), 
    ticks_per_step_(ticks_per_step), 
    obs_builder_(max_players_per_team, ticks_per_step, seed), 
    render_(render) {

    arena_ = render ? ffi::create_arena_vis(0) : ffi::create_arena(0);
    single_obs_size_ = obs_builder_.get_obs_size();
    action_space_size_ = action_parser_.get_action_space_size();

    // Fill teams.
    for (const ffi::Team team : match_layout) {
        add_agent(team);
    }

    // Construct state setter dist.
    std::vector<double> weights;
    weights.reserve(setters_.size());
    for (const auto& dist : setters_) {
        weights.push_back(dist.weight);
    }
    setter_selector_ = std::discrete_distribution<size_t>(weights.begin(), weights.end());

    // Setup Rewards.
    reward_manager_.set_rewards(rewards);

    // reward_manager_.add_reward("VelocityToBall", reward::create_velocity_to_ball_reward(), 0.1f);
    // reward_manager_.add_reward("TouchBall", reward::create_touch_ball_reward(), 1.0f);
    // reward_manager_.add_reward("Goal", reward::create_goal_reward(), 200.0f);
}

uint32_t RocketSimEnv::add_agent(ffi::Team team) {
    const uint32_t car_id = ffi::add_car(arena_, team);
    agents_.push_back({car_id, team});

    // Resize buffers for the new agent.
    obs_buffer_.resize(agents_.size() * single_obs_size_, 0.0f);
    action_mask_buffer_.resize(agents_.size() * action_space_size_, 0.0f);
    reward_buffer_.resize(agents_.size(), 0.0f);
    agent_x_inverted_.resize(agents_.size(), false);

    return car_id;
}

ResetResult RocketSimEnv::reset() {
    const size_t chosen_idx = setter_selector_(rng_);
    setters_[chosen_idx].setter->apply(arena_, arena_state_, agents_, rng_);
    reward_manager_.reset(arena_state_);
    obs_builder_.pre_step_rand(agents_);

    for (size_t i = 0; i < agents_.size(); i++) {
        float* agent_obs_ptr = obs_buffer_.data() + (i * single_obs_size_);
        agent_x_inverted_[i] = obs_builder_.build_obs(arena_state_, agents_, static_cast<uint32_t>(i), agent_obs_ptr);

        const ffi::CarState& car_state = arena_state_.cars[agents_[i].car_id];
        float* agent_action_mask_ptr = action_mask_buffer_.data() + (i * action_space_size_);
        action_parser_.get_action_mask(car_state, agent_action_mask_ptr);
    }

    return { obs_buffer_, action_mask_buffer_ };
}

StepResult RocketSimEnv::step(const int* actions) {
    for (size_t i = 0; i < agents_.size(); i++) {
        ffi::CarControls controls = decode_action(actions[i]);

        // Invert necessary controls if the state was X-mirrored.
        if (agent_x_inverted_[i]) {
            controls.steer  *= -1.0f;
            controls.yaw    *= -1.0f;
            controls.roll   *= -1.0f;
        }

        ffi::set_car_controls(arena_, agents_[i].car_id, controls);
    }

    std::fill(reward_buffer_.begin(), reward_buffer_.end(), 0.0f);
    int ticks_elapsed = ticks_per_step_;
    bool episode_terminated = false;
    bool episode_truncated = false;

    for (int i = 0; i < ticks_per_step_; i++) {
        ffi::step_arena(arena_);
        ffi::get_arena_state(arena_, arena_state_);

        if (render_) [[unlikely]] {
            std::this_thread::sleep_for(std::chrono::microseconds(8333));
        }

        // Calculate and accumulate rewards.
        for (size_t a = 0; a < agents_.size(); a++) {
            reward_buffer_[a] += reward_manager_.get_reward(agents_[a], arena_state_);
        }

        // Update previous state.
        reward_manager_.update_previous_state(arena_state_);

        // Reset touch counter if any car hit the ball or increment it.
        if (std::any_of(arena_state_.events.car_hit_ball, arena_state_.events.car_hit_ball + 8, [](bool v) { return v; })) {
            ticks_since_last_touch_ = 0;
        } else {
            ++ticks_since_last_touch_;
        }
        // Increment match ticks.
        ++match_ticks_;

        // Terminal state: Goal.
        if (arena_state_.events.is_ball_scored) {
            ticks_since_last_touch_ = 0;
            match_ticks_ = 0;
            ticks_elapsed = i + 1;
            episode_terminated = true;
            break;
        }

        // Tick count related truncation states.
        if (ticks_since_last_touch_ >= no_touch_ticks_limit_ || match_ticks_ >= match_ticks_limit_) {
            ticks_since_last_touch_ = 0;
            match_ticks_ = 0;
            ticks_elapsed = i + 1;
            episode_truncated = true;
            break;
        }
    }

    // Build the final observations.
    obs_builder_.pre_step_rand(agents_);
    for (size_t i = 0; i < agents_.size(); i++) {
        float* agent_obs_ptr = obs_buffer_.data() + (i * single_obs_size_);
        agent_x_inverted_[i] = obs_builder_.build_obs(arena_state_, agents_, static_cast<uint32_t>(i), agent_obs_ptr);

        const ffi::CarState& car_state = arena_state_.cars[agents_[i].car_id];
        float* agent_action_mask_ptr = action_mask_buffer_.data() + (i * action_space_size_);
        action_parser_.get_action_mask(car_state, agent_action_mask_ptr);
    }

    return { obs_buffer_, action_mask_buffer_, reward_buffer_, episode_terminated, episode_truncated, ticks_elapsed };
}

ffi::CarControls RocketSimEnv::decode_action(size_t action_idx) {
    return action_parser_.get_action(action_idx);
}

}; // namespace buta_ppo::env