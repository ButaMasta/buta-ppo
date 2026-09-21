// buta-ppo/src/env/vec_env.hpp
#pragma once

#include "rocketsim_env.hpp"
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <torch/torch.h>

namespace buta_ppo::env {

struct BatchedResetResult {
    torch::Tensor observations;
    torch::Tensor action_masks;
};

struct BatchedStepResult {
    torch::Tensor observations;
    torch::Tensor action_masks;
    torch::Tensor rewards;
    torch::Tensor dones;
};

class VecEnv {
private:
    size_t num_envs_;
    size_t agents_per_env_;
    size_t single_obs_size_;
    size_t action_space_size_;
    size_t total_agents_;

    std::vector<std::unique_ptr<RocketSimEnv>> envs_;

    torch::Tensor batched_obs_;
    torch::Tensor batched_action_masks_;
    torch::Tensor batched_rewards_;
    torch::Tensor batched_dones_;

    std::vector<std::thread> workers_;
    std::atomic<bool> terminate_pool_{false};
    std::atomic<int> pending_tasks_{0};

    std::atomic<int> batch_count_{0}; // Identifier for what batch number is expected of all workers.

    std::mutex start_mutex_;
    std::condition_variable cv_start_;

    std::mutex done_mutex_;
    std::condition_variable cv_done_;

    const int* current_actions_ptr_ = nullptr;
    enum class WorkerState { IDLE, RESET, STEP };
    WorkerState current_worker_state_ = WorkerState::IDLE;

    void worker_loop(size_t worker_id, size_t start_idx, size_t end_idx);
    
public:
    VecEnv(size_t num_envs, size_t num_threads, int ticks_per_step = 8, size_t max_players_per_team = 4);
    ~VecEnv();

    // NOTE: Do not read any rewards or dones after this call and before a step call as the data is stale.
    BatchedResetResult reset();

    BatchedStepResult step(const int* batched_actions);

    [[nodiscard]] size_t get_total_agents() const { return total_agents_; };
    [[nodiscard]] size_t get_single_obs_size() const { return single_obs_size_; };
    [[nodiscard]] size_t get_action_space_size() const { return action_space_size_; };
};

}; // namespace buta_ppo::env