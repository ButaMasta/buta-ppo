// buta-ppo/src/env/vec_env.hpp
#pragma once

#include "buta_ppo/ffi.h"
#include "state/state_setter.hpp"
#include "reward/reward_manager.hpp"
#include "metric/metric_manager.hpp"
#include "rl/pfsp.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>
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
 * @brief A contiguous range of batched rows.
 */
struct RowRange {
    size_t begin = 0;
    size_t count = 0;
};

/**
 * @brief A contiguous range of envs that can be stepped on its own, and the batched rows of its agents.
 *
 * Learner rows of all groups come first in the batched buffers, in group order, so each group's learner rows
 * are contiguous and the learner rows of every group together are rows [0, learner agents). The opponent rows
 * of PFSP envs follow, grouped by env group and then by opponent slot.
 */
struct EnvGroup {
    size_t env_begin = 0;
    size_t env_end = 0;
    size_t learner_begin = 0;
    size_t learner_end = 0;

    // This group's opponent rows for each opponent slot. Empty when PFSP is disabled.
    std::vector<RowRange> opponent_slots{};

    [[nodiscard]] size_t learner_count() const { return learner_end - learner_begin; }
};

/**
 * @brief How episodes ended across all envs.
 */
struct EpisodeCounts {
    uint32_t goals = 0;
    uint32_t no_touch_truncations = 0;   // No car touched the ball for too long.
    uint32_t time_limit_truncations = 0; // The match time limit ran out.
};

/**
 * @brief How expensive the envs were to step, summed across all envs.
 */
struct EnvCostCounts {
    uint64_t env_steps = 0;
    double step_seconds = 0.0;  // Worker time spent stepping envs, including resets and copying out the results.
    uint64_t touch_ticks = 0;   // Ticks in which any car touched the ball.
    uint64_t repredictions = 0; // Full ball predictions, mostly after touches.
};

/**
 * @brief Episode results of PFSP matches from the learner's side. Truncated episodes aren't counted.
 */
struct PFSPOutcome {
    uint32_t wins = 0;
    uint32_t losses = 0;
};

/**
 * @brief Manages the thread pool for multiple RocketSim envs in parallel and construct batches.
 *
 * An env's agents don't occupy contiguous rows of the batched buffers. Each agent has its own row, given by
 * `agent_rows_`, so the rows can be ordered by role (learner rows first) instead of by env.
 */
class VecEnv {
private:
    size_t num_envs_;
    std::vector<size_t> env_agent_counts_;
    size_t single_obs_size_;
    size_t action_space_size_;
    size_t total_agents_;
    size_t learner_agents_;

    // Batched row of each env's agents, in env order. Env i's agents start at `env_agent_offsets_[i]`.
    std::vector<uint32_t> agent_rows_;
    std::vector<size_t> env_agent_offsets_;

    // Per-env scratch for gathering an env's actions from its rows. Indexed like `agent_rows_`.
    std::vector<int> env_actions_;

    // PFSP state per env. Envs that aren't PFSP have no opponent team and a slot of -1.
    size_t num_opponent_slots_ = 0;
    std::vector<std::optional<ffi::Team>> env_opponent_team_;
    std::vector<int> env_opponent_slot_;
    // Written only by the worker stepping the env.
    std::vector<PFSPOutcome> env_outcomes_;
    std::vector<EpisodeCounts> env_episode_counts_;
    std::vector<EnvCostCounts> env_cost_counts_; // Only steps and step time, the env counts the rest.

    std::vector<std::unique_ptr<RocketSimEnv>> envs_;
    std::unordered_map<std::string, reward::RewardAccumulation> aggregate_reward_breakdown_;
    std::unordered_map<std::string, metric::MetricAccumulation> aggregate_metric_breakdown_;

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
     * @brief Scatters an env's per-agent obs and action masks into its agents' rows of the batched buffers.
     */
    void write_obs_and_masks(size_t env_idx, const std::vector<float>& obs, const std::vector<float>& masks);

    /**
     * @brief The rows of the batched step buffers belonging to a range of rows.
     */
    [[nodiscard]] BatchedStepResult step_result_rows(size_t row_begin, size_t row_count) const;

    
public:
    VecEnv(
        size_t num_envs,
        const std::vector<rl::MatchDistribution>& match_distributions,
        const std::vector<state::StateSetterDistribution>& setter_distributions,
        const std::vector<reward::RewardEntry>& reward_entries,
        const std::vector<metric::MetricEntry>& metric_entries,
        size_t num_threads, 
        int ticks_per_step = 8, 
        size_t max_players_per_team = 4, 
        bool render = false,
        const rl::PFSPConfig& pfsp_cfg = {},
        const std::vector<int>& worker_cpus = {}
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
     * @param batched_actions The actions for ALL rows. Only the rows of this group's agents are read.
     */
    void step_async(size_t group, const int* batched_actions);

    /**
     * @brief Blocks until the group started by `step_async` has finished stepping.
     *
     * @return That group's learner rows of the batched step buffers.
     */
    [[nodiscard]] BatchedStepResult wait();

    /**
     * @brief Sums how expensive the envs were to step since the last call, then clears the counts.
     *
     * Must not be called while an env group is stepping.
     */
    [[nodiscard]] EnvCostCounts collect_cost_counts();

    /**
     * @brief Switches every env to the benchmark's heavy workload, or back. See `RocketSimEnv::set_heavy_workload`.
     *
     * Must not be called while an env group is stepping.
     */
    void set_heavy_workload(bool enabled);

    /**
     * @brief Update the aggregated rewards and metrics for telemetry logging.
     * 
     * Clears the per env accumulators and throws an error if called when a group is stepping.
     */
    void update_telemetry_breakdowns();

    /**
     * @brief Sums the PFSP episode results per opponent slot since the last call, then clears them.
     *
     * Must not be called while an env group is stepping.
     *
     * @return The results of each opponent slot. Empty when PFSP is disabled.
     */
    [[nodiscard]] std::vector<PFSPOutcome> collect_pfsp_outcomes();

    /**
     * @brief Sums how episodes ended across all envs since the last call, then clears the counts.
     *
     * Must not be called while an env group is stepping.
     */
    [[nodiscard]] EpisodeCounts collect_episode_counts();

    [[nodiscard]] const std::unordered_map<std::string, reward::RewardAccumulation>& get_reward_breakdown() const { return aggregate_reward_breakdown_; };
    [[nodiscard]] const std::unordered_map<std::string, metric::MetricAccumulation>& get_metric_breakdown() const { return aggregate_metric_breakdown_; };
    [[nodiscard]] const std::vector<EnvGroup>& get_groups() const { return groups_; };
    // Every agent's row, which is the row count of the batched buffers and the batched actions.
    [[nodiscard]] size_t get_total_agents() const { return total_agents_; };
    // Agents controlled by the learning policy. Their rows are [0, learner agents).
    [[nodiscard]] size_t get_learner_agents() const { return learner_agents_; };
    [[nodiscard]] size_t get_num_opponent_slots() const { return num_opponent_slots_; };
    [[nodiscard]] size_t get_single_obs_size() const { return single_obs_size_; };
    [[nodiscard]] size_t get_action_space_size() const { return action_space_size_; };
};

} // namespace buta_ppo::env