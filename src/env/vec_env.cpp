// buta-ppo/src/env/vec_env.cpp

#include "buta_ppo/ffi.h"
#include "rocketsim_env.hpp"
#include "vec_env.hpp"
#include "rl/ppo_runner.hpp"

#include <cstddef>
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
    bool render
) {
    num_envs_ = num_envs_per_thread * num_threads;
    
    const auto env_counts = rl::compute_env_counts(num_envs_, match_distributions);

    envs_.reserve(num_envs_);
    env_agent_counts_.resize(num_envs_);
    env_agent_offsets_.resize(num_envs_);

    size_t current_agent_offset = 0;
    size_t env_idx = 0;

    for (size_t d = 0; d < match_distributions.size(); d++) {
        const rl::MatchDistribution& dist = match_distributions[d];
        const size_t count_for_dist = env_counts[d];
        const std::vector<ffi::Team> layout = dist.to_team_layout();
        const size_t agents_in_layout = dist.total_players();

        for (size_t c = 0; c < count_for_dist; c++) {
            envs_.push_back(std::make_unique<RocketSimEnv>(
                layout, 
                setter_distributions,
                reward_entries,
                ticks_per_step, 
                max_players_per_team, 
                std::random_device{}(), 
                render
            ));

            env_agent_counts_[env_idx] = agents_in_layout;
            env_agent_offsets_[env_idx] = current_agent_offset;

            current_agent_offset += agents_in_layout;
            env_idx++;
        }
    }

    total_agents_ = current_agent_offset;
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
        const size_t split_agent = env_agent_offsets_[split];
        groups_.push_back({0, split, 0, split_agent});
        groups_.push_back({split, num_envs_, split_agent, total_agents_});
    } else {
        groups_.push_back({0, num_envs_, 0, total_agents_});
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
                const size_t agent_offset = env_agent_offsets_[i];
                const auto& result = envs_[i]->reset();

                const size_t obs_offset = agent_offset * single_obs_size_;
                std::memcpy(batched_obs_.data_ptr<float>() + obs_offset, result.observations.data(), result.observations.size() * sizeof(float));

                const size_t action_mask_offset = agent_offset * action_space_size_;
                std::memcpy(batched_action_masks_.data_ptr<float>() + action_mask_offset, result.action_masks.data(), result.action_masks.size() * sizeof(float));
            }
            break;
        }
        case WorkerState::STEP: {
            // Claim envs until all are taken (see RESET).
            for (size_t i = next_env_.fetch_add(1, std::memory_order_relaxed); i < batch_env_end_;
                 i = next_env_.fetch_add(1, std::memory_order_relaxed)) {
                const size_t agent_offset = env_agent_offsets_[i];
                const size_t num_agents = env_agent_counts_[i];
                
                const int* env_actions_ptr = current_actions_ptr_ + agent_offset;
                StepResult result = envs_[i]->step(env_actions_ptr);

                const std::vector<float>* obs_src = &result.observations;
                const std::vector<float>* mask_src = &result.action_masks;

                if (result.is_terminated || result.is_truncated) {

                    // Route the final obs buffer to terminal obs.
                    if (result.is_truncated) {
                        const size_t obs_offset = agent_offset * single_obs_size_;
                        std::memcpy(batched_terminal_obs_.data_ptr<float>() + obs_offset,
                                    result.observations.data(),
                                    result.observations.size() * sizeof(float));
                    }

                    auto reset_res = envs_[i]->reset();
                    obs_src = &reset_res.observations;
                    mask_src = &reset_res.action_masks;
                }

                const size_t obs_offset = agent_offset * single_obs_size_;
                std::memcpy(batched_obs_.data_ptr<float>() + obs_offset, obs_src->data(), obs_src->size() * sizeof(float));

                const size_t action_mask_offset = agent_offset * action_space_size_;
                std::memcpy(batched_action_masks_.data_ptr<float>() + action_mask_offset, mask_src->data(), mask_src->size() * sizeof(float));

                const size_t reward_offset = agent_offset;
                std::memcpy(batched_rewards_.data_ptr<float>() + reward_offset, result.rewards.data(), result.rewards.size() * sizeof(float));

                const float term_val = result.is_terminated ? 1.0f : 0.0f;
                const float trunc_val = result.is_truncated ? 1.0f : 0.0f;
                for (size_t a = 0; a < num_agents; ++a) {
                    batched_terminated_.data_ptr<float>()[agent_offset + a] = term_val;
                    batched_truncated_.data_ptr<float>()[agent_offset + a] = trunc_val;
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

BatchedStepResult VecEnv::step_result_rows(size_t agent_begin, size_t agent_count) const {
    const auto begin = static_cast<int64_t>(agent_begin);
    const auto count = static_cast<int64_t>(agent_count);
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
    return step_result_rows(g.agent_begin, g.agent_count());
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