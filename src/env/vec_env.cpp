// buta-ppo/src/env/vec_env.cpp

#include "vec_env.hpp"
#include <algorithm>
#include <cstring>

namespace buta_ppo::env {

VecEnv::VecEnv(size_t num_envs, size_t num_threads, int ticks_per_step, size_t max_players_per_team, bool render)
    : num_envs_(num_envs) {
    
    for (size_t i = 0; i < num_envs_; i++) {
        auto env = std::make_unique<RocketSimEnv>(ticks_per_step, max_players_per_team, std::random_device{}(), render);

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
    action_space_size_ = envs_[0]->get_action_space_size();

    // Allocate Tensor Buffers
    auto pinned_opts = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);

    batched_obs_ = torch::zeros({(int64_t)total_agents_, (int64_t)single_obs_size_}, pinned_opts);
    batched_action_masks_ = torch::zeros({(int64_t)total_agents_, (int64_t)action_space_size_}, pinned_opts);
    batched_rewards_ = torch::zeros({(int64_t)total_agents_}, pinned_opts);
    batched_dones_ = torch::zeros({(int64_t)total_agents_}, pinned_opts);

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
                const auto& result = envs_[i]->reset();

                size_t obs_offset = i * agents_per_env_ * single_obs_size_;
                std::memcpy(batched_obs_.data_ptr<float>() + obs_offset, result.observations.data(), result.observations.size() * sizeof(float));

                size_t action_mask_offset = i * agents_per_env_ * action_space_size_;
                std::memcpy(batched_action_masks_.data_ptr<float>() + action_mask_offset, result.action_masks.data(), result.action_masks.size() * sizeof(float));
            }
            break;
        }
        case WorkerState::STEP: {
            for (size_t i = start_idx; i < end_idx; ++i) {
                
                const int* env_actions_ptr = current_actions_ptr_ + (i * agents_per_env_);

                StepResult result = envs_[i]->step(env_actions_ptr);

                const std::vector<float>* obs_src = &result.observations;
                const std::vector<float>* mask_src = &result.action_masks;

                if (result.is_done) {
                    auto reset_res = envs_[i]->reset();
                    obs_src = &reset_res.observations;
                    mask_src = &reset_res.action_masks;
                }

                size_t obs_offset = i * agents_per_env_ * single_obs_size_;
                std::memcpy(batched_obs_.data_ptr<float>() + obs_offset, obs_src->data(), obs_src->size() * sizeof(float));

                size_t action_mask_offset = i * agents_per_env_ * action_space_size_;
                std::memcpy(batched_action_masks_.data_ptr<float>() + action_mask_offset, mask_src->data(), mask_src->size() * sizeof(float));

                size_t reward_offset = i * agents_per_env_;
                std::memcpy(batched_rewards_.data_ptr<float>() + reward_offset, result.rewards.data(), result.rewards.size() * sizeof(float));

                float done_val = result.is_done ? 1.0f : 0.0f;
                for (size_t a = 0; a < agents_per_env_; ++a) {
                    batched_dones_.data_ptr<float>()[i * agents_per_env_ + a] = done_val;
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

BatchedResetResult VecEnv::reset() {
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
    return { batched_obs_, batched_action_masks_ };
}

BatchedStepResult VecEnv::step(const int* batched_actions) {
    {
        std::lock_guard<std::mutex> lock(start_mutex_);
        current_actions_ptr_ = batched_actions;
        current_worker_state_ = WorkerState::STEP;
        pending_tasks_ = workers_.size();
        batch_count_++; // New batch work is ready.
    }
    cv_start_.notify_all();

    std::unique_lock<std::mutex> lock(done_mutex_);
    cv_done_.wait(lock, [this] { return pending_tasks_ == 0; });

    current_worker_state_ = WorkerState::IDLE;
    current_actions_ptr_ = nullptr;

    return { batched_obs_, batched_action_masks_, batched_rewards_, batched_dones_ };
}

}; // namespace buta_ppo::env