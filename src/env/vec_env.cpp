// buta-ppo/src/env/vec_env.cpp

#include "vec_env.hpp"
#include <algorithm>
#include <cstring>

namespace buta_ppo::env {

VecEnv::VecEnv(size_t num_envs, size_t num_threads, int ticks_per_step, size_t max_players_per_team)
    : num_envs_(num_envs) {
    
    for (size_t i = 0; i < num_envs_; i++) {
        auto env = std::make_unique<RocketSimEnv>(ticks_per_step, max_players_per_team);

        // TODO: Account for more than just 1v1.
        // Starting with just 1v1.
        env->add_agent(ffi::Team::Blue);
        env->add_agent(ffi::Team::Orange);

        envs_.push_back(std::move(env));
    }

    // TODO: Account for uneven teams.
    // Assume all envs are symmetrical for now.
    agents_per_env_ = 2;
    total_agents_ = num_envs_ * agents_per_env_;
    single_obs_size_ = envs_[0]->get_obs_size();

    // Reserve space for buffers.
    batched_obs_.resize(total_agents_ * single_obs_size_, 0.0f);
    batched_rewards_.resize(total_agents_, 0.0f);
    batched_dones_.resize(num_envs_, 0);

    // Create thread pool.
    size_t actual_threads = std::min(num_threads, num_envs_);
    size_t envs_per_thread = num_envs_ / actual_threads;
    size_t remainder = num_envs_ % actual_threads;

    size_t current_start = 0;
    for (size_t i = 0; i < actual_threads; i++) {
        size_t chunk_size = envs_per_thread + (i < remainder ? 1 : 0);
        size_t current_end = current_start + chunk_size;

        workers_.emplace_back(&VecEnv::worker_loop, this, i, current_start, current_end);
        current_start = current_end;
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

void VecEnv::worker_loop([[maybe_unused]] size_t worker_id, size_t start_idx, size_t end_idx) {
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
            for (size_t i = start_idx; i < end_idx; i++) {
                const auto& env_obs = envs_[i]->reset();
                size_t mem_offset = i * agents_per_env_ * single_obs_size_;
                std::memcpy(&batched_obs_[mem_offset], env_obs.data(), env_obs.size() * sizeof(float));
            }
            break;
        }
        case WorkerState::STEP: {
            for (size_t i = start_idx; i < end_idx; ++i) {
                
                const int* env_actions_ptr = current_actions_ptr_ + (i * agents_per_env_);

                StepResult result = envs_[i]->step(env_actions_ptr);

                size_t obs_offset = i * agents_per_env_ * single_obs_size_;
                std::memcpy(&batched_obs_[obs_offset], result.observations.data(), result.observations.size() * sizeof(float));

                size_t reward_offset = i * agents_per_env_;
                std::memcpy(&batched_rewards_[reward_offset], result.rewards.data(), result.rewards.size() * sizeof(float));

                batched_dones_[i] = result.is_done ? 1 : 0;
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

const std::vector<float>& VecEnv::reset() {
    {
        std::lock_guard<std::mutex> lock(start_mutex_);
        current_worker_state_ = WorkerState::RESET;
        pending_tasks_ = workers_.size();
        batch_count_++; // New batch work is ready.
    }
    cv_start_.notify_all();

    std::unique_lock<std::mutex> lock(done_mutex_);
    cv_done_.wait(lock, [this] { return pending_tasks_ == 0; });

    current_worker_state_ = WorkerState::IDLE;
    return batched_obs_;
}

BatchedStepResult VecEnv::step(const std::vector<int>& batched_actions) {
    {
        std::lock_guard<std::mutex> lock(start_mutex_);
        current_actions_ptr_ = batched_actions.data();
        current_worker_state_ = WorkerState::STEP;
        pending_tasks_ = workers_.size();
        batch_count_++; // New batch work is ready.
    }
    cv_start_.notify_all();

    std::unique_lock<std::mutex> lock(done_mutex_);
    cv_done_.wait(lock, [this] { return pending_tasks_ == 0; });

    current_worker_state_ = WorkerState::IDLE;
    current_actions_ptr_ = nullptr;

    return { batched_obs_, batched_rewards_, batched_dones_ };
}

}; // namespace buta_ppo::env