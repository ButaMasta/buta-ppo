// buta-ppo/src/env/vec_env.cpp

#include "buta_ppo/ffi.h"
#include "rocketsim_env.hpp"
#include "vec_env.hpp"
#include "rl/ppo_runner.hpp"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace buta_ppo::env {

VecEnv::VecEnv(
    size_t num_envs_per_thread, 
    const std::vector<rl::MatchDistribution>& match_distributions, 
    const std::vector<state::StateSetterDistribution>& setter_distributions, 
    const std::vector<reward::RewardEntry>& reward_entries,
    size_t num_threads, 
    int ticks_per_step, 
    size_t max_players_per_team, 
    bool render,
    const rl::PFSPConfig& pfsp_cfg
) {
    num_envs_ = num_envs_per_thread * num_threads;

    // Rendering runs a single env, which is always self-play.
    const bool pfsp = pfsp_cfg.enabled() && !render;
    if (pfsp && (pfsp_cfg.envs >= 1.0f || pfsp_cfg.versions == 0)) {
        throw std::invalid_argument("PFSP envs must be in [0, 1) and PFSP versions at least 1.");
    }
    num_opponent_slots_ = pfsp ? pfsp_cfg.versions : 0;

    const auto env_counts = rl::compute_env_counts(num_envs_, match_distributions);

    envs_.reserve(num_envs_);
    env_agent_counts_.resize(num_envs_);
    env_agent_offsets_.resize(num_envs_);
    env_opponent_team_.resize(num_envs_);
    env_opponent_slot_.assign(num_envs_, -1);
    env_outcomes_.resize(num_envs_);

    // Whether each agent is controlled by a past policy, in env order like `agent_rows_`.
    std::vector<bool> agent_is_opponent;

    size_t current_agent_offset = 0;
    size_t env_idx = 0;
    size_t pfsp_env_count = 0;

    for (size_t d = 0; d < match_distributions.size(); d++) {
        const rl::MatchDistribution& dist = match_distributions[d];
        const size_t count_for_dist = env_counts[d];
        const std::vector<ffi::Team> layout = dist.to_team_layout();
        const size_t agents_in_layout = dist.total_players();

        // Only even teams play past versions.
        const bool even_teams = dist.blue_players == dist.orange_players;
        const size_t pfsp_for_dist = (pfsp && even_teams)
            ? static_cast<size_t>(std::lround(pfsp_cfg.envs * static_cast<float>(count_for_dist)))
            : 0;

        for (size_t c = 0; c < count_for_dist; c++) {
            // Spread the distribution's PFSP envs evenly over its envs, so both env groups get a similar share.
            const bool is_pfsp = (c + 1) * pfsp_for_dist / count_for_dist > c * pfsp_for_dist / count_for_dist;

            // Alternate the team the past version controls.
            std::optional<ffi::Team> opponent_team;
            if (is_pfsp) {
                opponent_team = (pfsp_env_count++ % 2 == 0) ? ffi::Team::Orange : ffi::Team::Blue;
            }

            envs_.push_back(std::make_unique<RocketSimEnv>(
                layout, 
                setter_distributions,
                reward_entries,
                ticks_per_step, 
                max_players_per_team, 
                std::random_device{}(), 
                render,
                opponent_team
            ));

            env_agent_counts_[env_idx] = agents_in_layout;
            env_agent_offsets_[env_idx] = current_agent_offset;
            env_opponent_team_[env_idx] = opponent_team;
            for (const ffi::Team team : layout) {
                agent_is_opponent.push_back(team == opponent_team);
            }

            current_agent_offset += agents_in_layout;
            env_idx++;
        }
    }

    if (pfsp && pfsp_env_count == 0) {
        throw std::invalid_argument("PFSP is enabled but no envs were given to it. "
                                    "It needs a match distribution with even teams and enough envs for the fraction.");
    }

    total_agents_ = current_agent_offset;
    agent_rows_.resize(total_agents_);
    env_actions_.resize(total_agents_);
    single_obs_size_ = envs_[0]->get_obs_size();
    action_space_size_ = envs_[0]->get_action_space_size();

    // Allocate Tensor Buffers
    const auto pinned_opts = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32).pinned_memory(true);

    batched_obs_ = torch::zeros({(int64_t)total_agents_, (int64_t)single_obs_size_}, pinned_opts);
    batched_action_masks_ = torch::zeros({(int64_t)total_agents_, (int64_t)action_space_size_}, pinned_opts);
    batched_rewards_ = torch::zeros({(int64_t)total_agents_}, pinned_opts);
    batched_terminated_ = torch::zeros({(int64_t)total_agents_}, pinned_opts);
    batched_truncated_ = torch::zeros({(int64_t)total_agents_}, pinned_opts);
    batched_terminal_obs_ = torch::zeros({(int64_t)total_agents_, (int64_t)single_obs_size_}, pinned_opts);

    // Split the envs into two groups with about half the agents each, on an env boundary, so one group's physics
    // can run while the other group's inference does. With a single env there is only one group.
    if (num_envs_ >= 2) {
        size_t split = 1;
        while (split < num_envs_ - 1 && env_agent_offsets_[split] < total_agents_ / 2) {
            split++;
        }
        groups_.push_back({0, split});
        groups_.push_back({split, num_envs_});
    } else {
        groups_.push_back({0, num_envs_});
    }

    // Deal each group's PFSP envs out to the opponent slots in turn.
    for (const EnvGroup& group : groups_) {
        size_t group_pfsp_envs = 0;
        for (size_t i = group.env_begin; i < group.env_end; i++) {
            if (env_opponent_team_[i]) {
                env_opponent_slot_[i] = static_cast<int>(group_pfsp_envs++ % num_opponent_slots_);
            }
        }
        if (group_pfsp_envs < num_opponent_slots_) {
            throw std::invalid_argument("Each env group needs at least as many PFSP envs as PFSP versions. "
                                        "Raise the PFSP env fraction or lower the PFSP versions.");
        }
    }

    // Assign every agent a batched row. First the learner rows of each group in group order.
    size_t next_row = 0;
    for (EnvGroup& group : groups_) {
        group.learner_begin = next_row;
        for (size_t i = group.env_begin; i < group.env_end; i++) {
            for (size_t a = 0; a < env_agent_counts_[i]; a++) {
                const size_t agent = env_agent_offsets_[i] + a;
                if (!agent_is_opponent[agent]) {
                    agent_rows_[agent] = static_cast<uint32_t>(next_row++);
                }
            }
        }
        group.learner_end = next_row;
    }
    learner_agents_ = next_row;

    // Then the opponent rows, by group and then by slot.
    for (EnvGroup& group : groups_) {
        group.opponent_slots.resize(num_opponent_slots_);
        for (size_t slot = 0; slot < num_opponent_slots_; slot++) {
            group.opponent_slots[slot].begin = next_row;
            for (size_t i = group.env_begin; i < group.env_end; i++) {
                if (env_opponent_slot_[i] != static_cast<int>(slot)) continue;
                for (size_t a = 0; a < env_agent_counts_[i]; a++) {
                    const size_t agent = env_agent_offsets_[i] + a;
                    if (agent_is_opponent[agent]) {
                        agent_rows_[agent] = static_cast<uint32_t>(next_row++);
                    }
                }
            }
            group.opponent_slots[slot].count = next_row - group.opponent_slots[slot].begin;
        }
    }

    // Create thread pool.
    for (size_t i = 0; i < num_threads; i++) {
        workers_.emplace_back(&VecEnv::worker_loop, this, i);
    }
}

VecEnv::~VecEnv() {
    {
        std::lock_guard<std::mutex> lock(start_mutex_);
        terminate_pool_ = true;
    }
    cv_start_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void VecEnv::worker_loop(size_t /*worker_id*/) {
    int local_batch_count = 0; // Track this worker's batch count.

    while (!terminate_pool_) {
        // Wait for main thread.
        {
            std::unique_lock<std::mutex> lock(start_mutex_);
            cv_start_.wait(lock, [this, &local_batch_count] {
                return (batch_count_ > local_batch_count) || terminate_pool_;
            });
        }

        if (terminate_pool_) break;

        local_batch_count = batch_count_;

        switch (current_worker_state_) {
        case WorkerState::RESET: {
            // Claim envs until all are taken. Relaxed is enough: the batch_count_/pending_tasks_ handshake
            // orders an env's writes by one worker before its next use by another.
            for (size_t i = next_env_.fetch_add(1, std::memory_order_relaxed); i < batch_env_end_;
                 i = next_env_.fetch_add(1, std::memory_order_relaxed)) {
                const auto& result = envs_[i]->reset();
                write_obs_and_masks(i, result.observations, result.action_masks);
            }
            break;
        }
        case WorkerState::STEP: {
            // Claim envs until all are taken (see RESET).
            for (size_t i = next_env_.fetch_add(1, std::memory_order_relaxed); i < batch_env_end_;
                 i = next_env_.fetch_add(1, std::memory_order_relaxed)) {
                const size_t agent_offset = env_agent_offsets_[i];
                const size_t num_agents = env_agent_counts_[i];
                const uint32_t* rows = agent_rows_.data() + agent_offset;

                // Gather the env's actions from its agents' rows.
                int* env_actions_ptr = env_actions_.data() + agent_offset;
                for (size_t a = 0; a < num_agents; ++a) {
                    env_actions_ptr[a] = current_actions_ptr_[rows[a]];
                }
                StepResult result = envs_[i]->step(env_actions_ptr);

                // Rewards and the final obs live in the env's buffers, which its reset overwrites, so copy them first.
                const float term_val = result.is_terminated ? 1.0f : 0.0f;
                const float trunc_val = result.is_truncated ? 1.0f : 0.0f;
                float* rewards = batched_rewards_.data_ptr<float>();
                float* terminated = batched_terminated_.data_ptr<float>();
                float* truncated = batched_truncated_.data_ptr<float>();
                for (size_t a = 0; a < num_agents; ++a) {
                    rewards[rows[a]] = result.rewards[a];
                    terminated[rows[a]] = term_val;
                    truncated[rows[a]] = trunc_val;
                }

                // A PFSP match's goal is a win when the learner's team, the one the past version doesn't control, scored.
                if (result.is_terminated && env_opponent_team_[i]) {
                    const bool learner_is_blue = *env_opponent_team_[i] == ffi::Team::Orange;
                    if (result.blue_scored == learner_is_blue) {
                        env_outcomes_[i].wins++;
                    } else {
                        env_outcomes_[i].losses++;
                    }
                }

                if (result.is_terminated || result.is_truncated) {

                    // Route the final obs buffer to terminal obs.
                    if (result.is_truncated) {
                        float* terminal_obs = batched_terminal_obs_.data_ptr<float>();
                        for (size_t a = 0; a < num_agents; ++a) {
                            std::memcpy(terminal_obs + rows[a] * single_obs_size_,
                                        result.observations.data() + a * single_obs_size_,
                                        single_obs_size_ * sizeof(float));
                        }
                    }

                    const auto& reset_res = envs_[i]->reset();
                    write_obs_and_masks(i, reset_res.observations, reset_res.action_masks);
                } else {
                    write_obs_and_masks(i, result.observations, result.action_masks);
                }
            }
            break;
        }
        default:
            break;
        }

        // If this was the last thread then thread operations are complete.
        if (--pending_tasks_ == 0) {
            std::lock_guard<std::mutex> lock(done_mutex_);
            cv_done_.notify_one();
        }
    }
}

void VecEnv::write_obs_and_masks(size_t env_idx, const std::vector<float>& obs, const std::vector<float>& masks) {
    const uint32_t* rows = agent_rows_.data() + env_agent_offsets_[env_idx];
    float* batched_obs = batched_obs_.data_ptr<float>();
    float* batched_masks = batched_action_masks_.data_ptr<float>();

    for (size_t a = 0; a < env_agent_counts_[env_idx]; ++a) {
        std::memcpy(batched_obs + rows[a] * single_obs_size_, obs.data() + a * single_obs_size_,
                    single_obs_size_ * sizeof(float));
        std::memcpy(batched_masks + rows[a] * action_space_size_, masks.data() + a * action_space_size_,
                    action_space_size_ * sizeof(float));
    }
}

void VecEnv::dispatch(WorkerState state, const int* batched_actions, size_t env_begin, size_t env_end) {
    {
        std::lock_guard<std::mutex> lock(start_mutex_);
        current_actions_ptr_ = batched_actions;
        current_worker_state_ = state;
        next_env_ = env_begin;
        batch_env_end_ = env_end;
        pending_tasks_ = workers_.size();
        batch_count_++; // New batch work is ready.
    }
    cv_start_.notify_all();
}

void VecEnv::wait_for_workers() {
    {
        std::unique_lock<std::mutex> lock(done_mutex_);
        cv_done_.wait(lock, [this] { return pending_tasks_ == 0; });
    }

    current_worker_state_ = WorkerState::IDLE;
    current_actions_ptr_ = nullptr;
}

BatchedStepResult VecEnv::step_result_rows(size_t row_begin, size_t row_count) const {
    const auto begin = static_cast<int64_t>(row_begin);
    const auto count = static_cast<int64_t>(row_count);
    return {
        batched_obs_.narrow(0, begin, count),
        batched_action_masks_.narrow(0, begin, count),
        batched_rewards_.narrow(0, begin, count),
        batched_terminated_.narrow(0, begin, count),
        batched_truncated_.narrow(0, begin, count),
        batched_terminal_obs_.narrow(0, begin, count)
    };
}

BatchedResetResult VecEnv::reset() {
    if (group_in_flight_) {
        throw std::logic_error("VecEnv::reset called while an env group is still stepping.");
    }
    dispatch(WorkerState::RESET, nullptr, 0, num_envs_);
    wait_for_workers();
    return { batched_obs_, batched_action_masks_ };
}

BatchedStepResult VecEnv::step(const int* batched_actions) {
    if (group_in_flight_) {
        throw std::logic_error("VecEnv::step called while an env group is still stepping.");
    }
    dispatch(WorkerState::STEP, batched_actions, 0, num_envs_);
    wait_for_workers();
    return { batched_obs_, batched_action_masks_, batched_rewards_, batched_terminated_, batched_truncated_, batched_terminal_obs_ };
}

void VecEnv::step_async(size_t group, const int* batched_actions) {
    if (group_in_flight_) {
        throw std::logic_error("VecEnv::step_async called while another env group is still stepping.");
    }
    const EnvGroup& g = groups_.at(group);
    dispatch(WorkerState::STEP, batched_actions, g.env_begin, g.env_end);
    group_in_flight_ = true;
    in_flight_group_ = group;
}

BatchedStepResult VecEnv::wait() {
    if (!group_in_flight_) {
        throw std::logic_error("VecEnv::wait called with no env group stepping.");
    }
    wait_for_workers();
    group_in_flight_ = false;

    const EnvGroup& g = groups_[in_flight_group_];
    return step_result_rows(g.learner_begin, g.learner_count());
}

std::vector<PFSPOutcome> VecEnv::collect_pfsp_outcomes() {
    if (group_in_flight_) {
        throw std::logic_error("VecEnv::collect_pfsp_outcomes called while an env group is still stepping.");
    }

    std::vector<PFSPOutcome> slot_outcomes(num_opponent_slots_);
    for (size_t i = 0; i < num_envs_; i++) {
        if (env_opponent_slot_[i] < 0) continue;
        PFSPOutcome& slot = slot_outcomes[static_cast<size_t>(env_opponent_slot_[i])];
        slot.wins += env_outcomes_[i].wins;
        slot.losses += env_outcomes_[i].losses;
        env_outcomes_[i] = {};
    }
    return slot_outcomes;
}

void VecEnv::update_reward_breakdown() {
    for (auto& [name, val] : aggregate_reward_breakdown_) {
        val = 0.0;
    }

    for (const auto& env : envs_) {
        env->update_reward_telemetry();
        const auto& env_breakdown = env->get_reward_telemetry();

        for (const auto& [name, val] : env_breakdown) {
            aggregate_reward_breakdown_[name] += val;
        }
    }
}

} // namespace buta_ppo::env