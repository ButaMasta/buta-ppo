// buta-ppo/src/env/vec_env.hpp
#pragma once

#include "state/state_setter.hpp"
#include "reward/reward_manager.hpp"

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
    torch::Tensor terminated;
    torch::Tensor truncated;
    torch::Tensor terminal_observations;
};

/**
 * @brief A contiguous range of envs and their agents that can be stepped on its own.
 */
struct EnvGroup {
    size_t env_begin = 0;
    size_t env_end = 0;
    size_t agent_begin = 0;
    size_t agent_end = 0;

    [[nodiscard]] size_t agent_count() const { return agent_end - agent_begin; }
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
    torch::Tensor batched_terminated_;
    torch::Tensor batched_truncated_;
    torch::Tensor batched_terminal_obs_;


    std::vector<std::thread> workers_;
    std::atomic<bool> terminate_pool_{false};
    std::atomic<int> pending_tasks_{0};

    // Identifier for what batch number is expected of all workers.
    std::atomic<int> batch_count_{0};

    // Index of the next env to be claimed by a worker in the current batch. Set to the batch's first env.
    std::atomic<size_t> next_env_{0};
    // One past the last env of the current batch. Written by the main thread before each batch.
    size_t batch_env_end_ = 0;

    // Env groups for pipelined stepping. Two groups when there are at least 2 envs, otherwise one.
    std::vector<EnvGroup> groups_;
    bool group_in_flight_ = false;
    size_t in_flight_group_ = 0;

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
     * Each batch, workers claim envs one at a time from a shared counter until all are taken,
     * so faster or earlier-woken threads take on more of the work.
     *
     * @param worker_id The ID of this thread.
     */
    void worker_loop(size_t worker_id);

    /**
     * @brief Wakes the workers to process envs [env_begin, env_end) in the given state. Returns immediately.
     */
    void dispatch(WorkerState state, const int* batched_actions, size_t env_begin, size_t env_end);

    /**
     * @brief Blocks until the workers have finished the dispatched batch.
     */
    void wait_for_workers();

    /**
     * @brief The rows of the batched step buffers belonging to a range of agents.
     */
    [[nodiscard]] BatchedStepResult step_result_rows(size_t agent_begin, size_t agent_count) const;

    
public:
    VecEnv(
        size_t num_envs_per_thread, 
        const std::vector<rl::MatchDistribution>& match_distributions,
        const std::vector<state::StateSetterDistribution>& setter_distributions,
        const std::vector<reward::RewardEntry>& reward_entries,
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
     * @brief Step every env and retrieve the batched result.
     * 
     * @param batched_actions The actions to dispatch to the workers.
     * @return BatchedStepResult The entire result of all worker's step.
     */
    [[nodiscard]] BatchedStepResult step(const int* batched_actions);

    /**
     * @brief Starts stepping one env group and returns immediately, so the caller can work while it runs.
     *
     * Only one group can be in flight at a time. The caller must not touch the group's rows of the batched
     * buffers until `wait` returns. Other groups' rows are safe to read and write in the meantime.
     *
     * @param group The group to step.
     * @param batched_actions The actions for ALL agents. Only this group's entries are read.
     */
    void step_async(size_t group, const int* batched_actions);

    /**
     * @brief Blocks until the group started by `step_async` has finished stepping.
     *
     * @return That group's rows of the batched step buffers.
     */
    [[nodiscard]] BatchedStepResult wait();

    /**
     * @brief Update the aggregated rewards for telemetry logging.
     */
    void update_reward_breakdown();

    [[nodiscard]] const std::unordered_map<std::string, double>& get_reward_breakdown() const { return aggregate_reward_breakdown_; };
    [[nodiscard]] const std::vector<EnvGroup>& get_groups() const { return groups_; };
    [[nodiscard]] size_t get_total_agents() const { return total_agents_; };
    [[nodiscard]] size_t get_single_obs_size() const { return single_obs_size_; };
    [[nodiscard]] size_t get_action_space_size() const { return action_space_size_; };
};

} // namespace buta_ppo::env