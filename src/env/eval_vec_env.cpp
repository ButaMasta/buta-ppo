// buta-ppo/src/env/eval_vec_env.cpp
#include "eval_vec_env.hpp"
#include "rocketsim_env.hpp"
#include "episode_limits.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace buta_ppo::env {

EvalVecEnv::EvalVecEnv(
    std::vector<EvalGame> games,
    size_t num_anchor_slots,
    int ticks_per_step,
    int game_cap_ticks,
    size_t max_players_per_team,
    size_t num_threads,
    const std::vector<int>& worker_cpus
) : ticks_per_step_(ticks_per_step), pool_(num_threads, worker_cpus) {
    if (games.empty() || num_threads == 0 || game_cap_ticks <= 0) {
        throw std::invalid_argument("EvalVecEnv needs at least one game, one worker thread and a positive game cap.");
    }

    // No no-touch limit: like in game, a game only ends on a goal, or as a draw at the cap.
    const EpisodeLimits limits{0, game_cap_ticks};

    envs_.reserve(games.size());
    size_t agent_offset = 0;
    for (size_t g = 0; g < games.size(); g++) {
        EvalGame& game = games[g];
        if (game.team_size == 0 || game.team_size > max_players_per_team) {
            throw std::invalid_argument("EvalVecEnv: game " + std::to_string(g) + " has a team size of "
                                        + std::to_string(game.team_size) + ", which must be in [1, max_players_per_team].");
        }
        if (!game.opponent_bot && game.anchor_slot >= num_anchor_slots) {
            throw std::invalid_argument("EvalVecEnv: game " + std::to_string(g) + " plays anchor slot "
                                        + std::to_string(game.anchor_slot) + " of only " + std::to_string(num_anchor_slots) + ".");
        }
        if (!game.setter) {
            throw std::invalid_argument("EvalVecEnv: game " + std::to_string(g) + " has no state setter.");
        }

        // Blue cars first, like `MatchDistribution::to_team_layout`.
        std::vector<ffi::Team> layout(game.team_size, ffi::Team::Blue);
        layout.insert(layout.end(), game.team_size, ffi::Team::Orange);
        const ffi::Team anchor_team = game.learner_team == ffi::Team::Blue ? ffi::Team::Orange : ffi::Team::Blue;

        // External groups in this order: the learner's side's bot, then the opponent's.
        std::vector<external::ExternalSide> external_sides;
        if (game.learner_bot) external_sides.push_back({game.learner_bot, game.learner_team});
        if (game.opponent_bot) external_sides.push_back({game.opponent_bot, anchor_team});

        std::vector<state::StateSetterDistribution> setters;
        setters.emplace_back(std::move(game.setter), 1.0f);

        envs_.push_back(std::make_unique<RocketSimEnv>(
            layout,
            setters,
            std::vector<reward::RewardEntry>{},
            std::vector<metric::MetricEntry>{},
            ticks_per_step,
            limits,
            max_players_per_team,
            game.seed,
            false,
            anchor_team,
            external_sides
        ));
        learner_teams_.push_back(game.learner_team);
        seeds_.push_back(game.seed);
        external_.push_back(!external_sides.empty());
        has_learner_bot_.push_back(game.learner_bot != nullptr);
        env_agent_offsets_.push_back(agent_offset);
        env_agent_counts_.push_back(layout.size());
        agent_offset += layout.size();
    }

    agent_rows_.assign(agent_offset, NO_ROW);
    env_actions_.resize(agent_offset);
    single_obs_size_ = envs_[0]->get_obs_size();
    action_space_size_ = envs_[0]->get_action_space_size();

    // An env's agents are its blue cars, then its orange cars.
    const auto is_learner = [&](size_t env_idx, size_t agent) {
        const ffi::Team team = agent < games[env_idx].team_size ? ffi::Team::Blue : ffi::Team::Orange;
        return team == learner_teams_[env_idx];
    };

    // Learner rows first, in game order.
    size_t next_row = 0;
    learner_rows_before_.reserve(envs_.size() + 1);
    for (size_t i = 0; i < envs_.size(); i++) {
        learner_rows_before_.push_back(next_row);
        if (games[i].learner_bot) continue;
        for (size_t a = 0; a < env_agent_counts_[i]; a++) {
            if (is_learner(i, a)) agent_rows_[env_agent_offsets_[i] + a] = static_cast<uint32_t>(next_row++);
        }
    }
    learner_rows_before_.push_back(next_row);
    learner_agents_ = next_row;

    // Then each anchor slot's rows, in game order within the slot.
    anchor_rows_.resize(num_anchor_slots);
    anchor_row_begin_.assign(envs_.size(), next_row);
    anchor_row_count_.assign(envs_.size(), 0);
    for (size_t slot = 0; slot < num_anchor_slots; slot++) {
        anchor_rows_[slot].begin = next_row;
        for (size_t i = 0; i < envs_.size(); i++) {
            if (games[i].opponent_bot || games[i].anchor_slot != slot) continue;
            anchor_row_begin_[i] = next_row;
            for (size_t a = 0; a < env_agent_counts_[i]; a++) {
                if (!is_learner(i, a)) agent_rows_[env_agent_offsets_[i] + a] = static_cast<uint32_t>(next_row++);
            }
            anchor_row_count_[i] = next_row - anchor_row_begin_[i];
        }
        anchor_rows_[slot].count = next_row - anchor_rows_[slot].begin;
    }
    total_agents_ = next_row;

    // One batch runner per external model, sized for every car of that model deciding at once, and a slot per model
    // and car count.
    std::vector<const void*> models;
    std::vector<std::map<size_t, size_t>> model_max_batch;
    std::vector<external::ExternalBot*> model_bots;
    std::map<std::pair<size_t, size_t>, size_t> slot_of; // (model, car count) -> slot
    group_slots_.resize(envs_.size());
    sized_decisions_.resize(envs_.size());
    group_models_.resize(envs_.size());
    for (size_t i = 0; i < envs_.size(); i++) {
        const size_t num_cars = env_agent_counts_[i];
        for (const auto* bot : {games[i].learner_bot.get(), games[i].opponent_bot.get()}) {
            if (!bot) continue;
            const auto found = std::find(models.begin(), models.end(), bot->model());
            const size_t model = static_cast<size_t>(found - models.begin());
            if (found == models.end()) {
                models.push_back(bot->model());
                model_max_batch.emplace_back();
                model_bots.push_back(const_cast<external::ExternalBot*>(bot));
            }
            // The most decisions the group can ask for in a round.
            const size_t most = envs_[i]->lockstep_max_decisions(group_slots_[i].size());
            sized_decisions_[i].push_back(most);
            model_max_batch[model][num_cars] += most;
            const auto [it, added] = slot_of.try_emplace({model, num_cars}, slots_.size());
            if (added) {
                slots_.push_back({nullptr, num_cars, bot->obs_size(num_cars), bot->decision_size(), 0});
            }
            group_slots_[i].push_back(it->second);
            group_models_[i].push_back(bot->model());
        }
    }
    for (size_t m = 0; m < models.size(); m++) {
        runners_.push_back(model_bots[m]->make_batch_runner(model_max_batch[m]));
    }

    for (const auto& [key, slot] : slot_of) {
        slots_[slot].runner = runners_[key.first].get();
    }
    size_t groups = 0;
    for (const auto& g : group_slots_) groups += g.size();
    gathered_.reserve(groups);

    outcomes_.assign(envs_.size(), GameOutcome::Unfinished);
    game_ticks_.assign(envs_.size(), 0);
    frozen_left_.assign(envs_.size(), 0);
    step_done_.assign(envs_.size(), 0);
    needs_inference_.assign(envs_.size(), 0);

    const auto pinned_opts = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32).pinned_memory(true);
    const auto rows = static_cast<int64_t>(std::max<size_t>(total_agents_, 1));
    batched_obs_ = torch::zeros({rows, (int64_t)single_obs_size_}, pinned_opts);
    batched_action_masks_ = torch::zeros({rows, (int64_t)action_space_size_}, pinned_opts);

    reset_task_ = [this](size_t i) {
        envs_[i]->reseed(seeds_[i]);
        const auto& result = envs_[i]->reset();
        write_obs_and_masks(i, result.observations, result.action_masks);
    };
    step_task_ = [this](size_t i) {
        if (outcomes_[i] == GameOutcome::Unfinished) step_env(i);
    };

    // Lockstep play, for a range with external bots. Envs without one play as above, a whole step at a time.
    lockstep_reset_task_ = [this](size_t i) {
        if (!external_[i]) {
            reset_task_(i);
            return;
        }
        envs_[i]->reseed(seeds_[i]);
        frozen_left_[i] = envs_[i]->lockstep_reset();
        needs_inference_[i] = 1;
    };
    frozen_task_ = [this](size_t i) {
        if (frozen_left_[i] == 0) return;
        const int packets = std::min(frozen_left_[i], envs_[i]->lockstep_packets());
        frozen_left_[i] -= packets;
        needs_inference_[i] = 1;
        if (const auto result = envs_[i]->lockstep_frozen(packets)) {
            write_obs_and_masks(i, result->observations, result->action_masks);
        }
    };
    tick_task_ = [this](size_t i) {
        if (outcomes_[i] != GameOutcome::Unfinished) return;
        if (!external_[i]) {
            if (lockstep_tick_ == 0) step_env(i);
            return;
        }
        if (lockstep_tick_ == 0) {
            envs_[i]->lockstep_begin_step(gather_actions(i));
            step_done_[i] = 0;
        }
        if (step_done_[i]) return;
        needs_inference_[i] = 1;
        if (const auto result = envs_[i]->lockstep_ticks(round_ticks_)) {
            step_done_[i] = 1;
            record_step(i, *result);
        }
    };
}

// Defined here, where RocketSimEnv is a complete type.
EvalVecEnv::~EvalVecEnv() = default;

void EvalVecEnv::set_external_bot(size_t game, bool learner_side, std::shared_ptr<external::ExternalBot> bot) {
    // External groups are the learner's side's bot, then the opponent's.
    const size_t groups = group_slots_.at(game).size();
    const bool has = learner_side ? has_learner_bot_[game] : groups > static_cast<size_t>(has_learner_bot_[game]);
    if (!has) {
        throw std::invalid_argument("EvalVecEnv::set_external_bot: game " + std::to_string(game) + " has no external bot on that side.");
    }
    const size_t group = learner_side ? 0 : groups - 1;
    if (!bot || bot->model() != group_models_[game][group]) {
        throw std::invalid_argument("EvalVecEnv::set_external_bot needs a bot of the same model, whose batch runner plays it.");
    }
    envs_[game]->set_external_bot(group, std::move(bot));
}

void EvalVecEnv::reset(size_t begin, size_t end) {
    end = std::min(end, envs_.size());
    if (begin >= end) {
        throw std::invalid_argument("EvalVecEnv::reset needs at least one game to play.");
    }
    std::fill(outcomes_.begin(), outcomes_.end(), GameOutcome::Unfinished);
    std::fill(game_ticks_.begin(), game_ticks_.end(), 0);

    played_begin_ = begin;
    played_end_ = end;
    range_external_ = std::any_of(external_.begin() + static_cast<std::ptrdiff_t>(begin),
                                  external_.begin() + static_cast<std::ptrdiff_t>(end), [](uint8_t e) { return e != 0; });
    if (!range_external_) {
        pool_.dispatch(reset_task_, played_begin_, played_end_);
        pool_.wait();
    } else {
        pool_.dispatch(lockstep_reset_task_, played_begin_, played_end_);
        pool_.wait();
        // Each round plays as many packets as every game's bots can wait for their decisions. Fresh controllers (e.g.
        // of a bot swapped in) must fit the runners, which were sized for a round's decisions.
        window_ = INT32_MAX;
        for (size_t i = played_begin_; i < played_end_; i++) {
            if (!external_[i]) continue;
            for (size_t group = 0; group < sized_decisions_[i].size(); group++) {
                if (envs_[i]->lockstep_max_decisions(group) > sized_decisions_[i][group]) {
                    throw std::logic_error("EvalVecEnv: game " + std::to_string(i) + "'s bots ask for more decisions per "
                                           "round than its batch runner was sized for.");
                }
            }
            window_ = std::min(window_, envs_[i]->lockstep_window());
        }
        // The frozen packets, all games in lockstep, until every game has played its own.
        for (;;) {
            bool frozen = false;
            for (size_t i = played_begin_; i < played_end_; i++) frozen |= frozen_left_[i] > 0;
            if (!frozen) break;
            infer_external();
            pool_.dispatch(frozen_task_, played_begin_, played_end_);
            pool_.wait();
        }
    }
    games_left_ = played_end_ - played_begin_;
}

void EvalVecEnv::step(const int* batched_actions) {
    if (all_done()) {
        throw std::logic_error("EvalVecEnv::step called with no unfinished games. Call reset first.");
    }

    // Published to the workers by the pool's dispatch.
    current_actions_ptr_ = batched_actions;
    if (!range_external_) {
        pool_.dispatch(step_task_, played_begin_, played_end_);
        pool_.wait();
    } else {
        // A window of ticks at a time, each after the external bots' decisions from the last, until every game's step
        // has ended.
        for (lockstep_tick_ = 0; lockstep_tick_ < ticks_per_step_; lockstep_tick_ += round_ticks_) {
            round_ticks_ = std::min(window_, ticks_per_step_ - lockstep_tick_);
            infer_external();
            pool_.dispatch(tick_task_, played_begin_, played_end_);
            pool_.wait();

            bool stepping = false;
            for (size_t i = played_begin_; i < played_end_; i++) {
                stepping |= external_[i] && outcomes_[i] == GameOutcome::Unfinished && !step_done_[i];
            }
            if (!stepping) break;
        }
    }
    current_actions_ptr_ = nullptr;

    const auto played_first = outcomes_.begin() + static_cast<std::ptrdiff_t>(played_begin_);
    const auto played_last = outcomes_.begin() + static_cast<std::ptrdiff_t>(played_end_);
    games_left_ = static_cast<size_t>(std::count(played_first, played_last, GameOutcome::Unfinished));
}

const int* EvalVecEnv::gather_actions(size_t env_idx) {
    const size_t agent_offset = env_agent_offsets_[env_idx];
    const size_t num_agents = env_agent_counts_[env_idx];
    const uint32_t* rows = agent_rows_.data() + agent_offset;

    int* env_actions_ptr = env_actions_.data() + agent_offset;
    for (size_t a = 0; a < num_agents; ++a) {
        env_actions_ptr[a] = rows[a] == NO_ROW ? 0 : current_actions_ptr_[rows[a]];
    }
    return env_actions_ptr;
}

void EvalVecEnv::step_env(size_t env_idx) {
    const StepResult result = envs_[env_idx]->step(gather_actions(env_idx));
    record_step(env_idx, result);
}

void EvalVecEnv::record_step(size_t env_idx, const StepResult& result) {
    game_ticks_[env_idx] += result.ticks_elapsed;

    if (result.is_terminated) {
        const bool learner_is_blue = learner_teams_[env_idx] == ffi::Team::Blue;
        outcomes_[env_idx] = result.blue_scored == learner_is_blue ? GameOutcome::Win : GameOutcome::Loss;
    } else if (result.is_truncated) {
        // The only limit is the game cap.
        outcomes_[env_idx] = GameOutcome::Draw;
    } else {
        write_obs_and_masks(env_idx, result.observations, result.action_masks);
    }
}

void EvalVecEnv::infer_external() {
    gathered_.clear();
    for (InferenceSlot& slot : slots_) slot.count = 0;

    // Gather, in game order, so the batches are the same every time the same games are played.
    for (size_t i = played_begin_; i < played_end_; i++) {
        if (!needs_inference_[i]) continue;
        needs_inference_[i] = 0;
        if (outcomes_[i] != GameOutcome::Unfinished) continue;
        for (size_t group = 0; group < group_slots_[i].size(); group++) {
            const RocketSimEnv::ExternalBatch batch = envs_[i]->external_batch(group);
            if (batch.count == 0) continue;
            InferenceSlot& slot = slots_[group_slots_[i][group]];
            std::memcpy(slot.runner->obs(slot.num_cars) + slot.count * slot.obs_size, batch.obs,
                        batch.count * slot.obs_size * sizeof(float));
            gathered_.push_back({i, group, group_slots_[i][group], slot.count});
            slot.count += batch.count;
        }
    }

    // Every slot is launched before any is waited for, so they can run at the same time.
    for (InferenceSlot& slot : slots_) {
        if (slot.count > 0) slot.runner->launch(slot.num_cars, slot.count);
    }
    std::vector<const float*>& decisions = slot_decisions_;
    decisions.assign(slots_.size(), nullptr);
    for (size_t s = 0; s < slots_.size(); s++) {
        if (slots_[s].count > 0) decisions[s] = slots_[s].runner->wait(slots_[s].num_cars);
    }
    for (const Gathered& g : gathered_) {
        const RocketSimEnv::ExternalBatch batch = envs_[g.env]->external_batch(g.group);
        const InferenceSlot& slot = slots_[g.slot];
        std::memcpy(batch.decisions, decisions[g.slot] + g.offset * slot.decision_size,
                    batch.count * slot.decision_size * sizeof(float));
    }
}

void EvalVecEnv::write_obs_and_masks(size_t env_idx, const std::vector<float>& obs, const std::vector<float>& masks) {
    const uint32_t* rows = agent_rows_.data() + env_agent_offsets_[env_idx];
    float* batched_obs = batched_obs_.data_ptr<float>();
    float* batched_masks = batched_action_masks_.data_ptr<float>();

    for (size_t a = 0; a < env_agent_counts_[env_idx]; ++a) {
        if (rows[a] == NO_ROW) continue;
        std::memcpy(batched_obs + rows[a] * single_obs_size_, obs.data() + a * single_obs_size_,
                    single_obs_size_ * sizeof(float));
        std::memcpy(batched_masks + rows[a] * action_space_size_, masks.data() + a * action_space_size_,
                    action_space_size_ * sizeof(float));
    }
}

} // namespace buta_ppo::env
