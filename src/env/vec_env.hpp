// buta-ppo/src/env/vec_env.hpp
#pragma once

#include "state_setter.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <torch/torch.h>

// Forward declarations.
namespace buta_ppo::rl  { struct MatchDistribution; }
namespace buta_ppo::env { class RocketSimEnv; }

namespace buta_ppo::env {

/**
 * @brief Batched result from resetting envs.
 */
struct BatchedResetResult {
    torch::Tensor observations;
    torch::Tensor action_masks;
};

/**
 * @brief Batched result for stepping envs.
 */
struct BatchedStepResult {
    torch::Tensor observations;
    torch::Tensor action_masks;
    torch::Tensor rewards;
    torch::Tensor dones;
};

/**
 * @brief Manages the thread pool for multiple RocketSim envs in parallel and construct batches.
 */
class VecEnv {
private:
    size_t num_envs_;
    std::vector<size_t> env_agent_counts_;
    std::vector<size_t> env_agent_offsets_;
    size_t single_obs_size_;
    size_t action_space_size_;
    size_t total_agents_;

    std::vector<std::unique_ptr<RocketSimEnv>> envs_;
    std::unordered_map<std::string, double> aggregate_reward_breakdown_;

    torch::Tensor batched_obs_;
    torch::Tensor batched_action_masks_;
    torch::Tensor batched_rewards_;
    torch::Tensor batched_dones_;

    std::vector<std::thread> workers_;
    std::atomic<bool> terminate_pool_{false};
    std::atomic<int> pending_tasks_{0};

    // Identifier for what batch number is expected of all workers.
    std::atomic<int> batch_count_{0};

    std::mutex start_mutex_;
    std::condition_variable cv_start_;

    std::mutex done_mutex_;
    std::condition_variable cv_done_;

    const int* current_actions_ptr_ = nullptr;
    enum class WorkerState { IDLE, RESET, STEP };
    WorkerState current_worker_state_ = WorkerState::IDLE;

    /**
     * @brief The multi-threaded method that all threads spawned run.
     * 
     * @param worker_id The ID of this thread.
     * @param start_idx The start idx of this worker's working area. (Which envs it is responsible for)
     * @param end_idx The end idx of this worker's working area.
     */
    void worker_loop(size_t worker_id, size_t start_idx, size_t end_idx);
    
public:
    VecEnv(
        size_t num_envs, 
        const std::vector<rl::MatchDistribution>& match_distributions,
        const std::vector<StateSetterDistribution>& setter_distributions,
        size_t num_threads, 
        int ticks_per_step = 8, 
        size_t max_players_per_team = 4, 
        bool render = false
    );

    ~VecEnv();

    /**
     * @brief Reset the vectorized environements.
     * 
     * NOTE: Do not read any rewards or dones after this call and before a step call as the data is stale.
     * 
     * @return BatchedResetResult The result of the reset.
     */
    [[nodiscard]] BatchedResetResult reset();

    /**
     * @brief Step the entrire worker threadpool and retrieve the batched result.
     * 
     * @param batched_actions The actions to dispatch to the workers.
     * @return BatchedStepResult The entire result of all worker's step.
     */
    [[nodiscard]] BatchedStepResult step(const int* batched_actions);

    /**
     * @brief Update the aggregated rewards for telemetry logging.
     */
    void update_reward_breakdown();

    [[nodiscard]] const std::unordered_map<std::string, double>& get_reward_breakdown() const { return aggregate_reward_breakdown_; };
    [[nodiscard]] size_t get_total_agents() const { return total_agents_; };
    [[nodiscard]] size_t get_single_obs_size() const { return single_obs_size_; };
    [[nodiscard]] size_t get_action_space_size() const { return action_space_size_; };
};

} // namespace buta_ppo::env