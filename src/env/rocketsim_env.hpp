// buta-ppo/src/env/rocketsim_env.hpp
#pragma once

#include "buta_ppo/ffi.h"
#include "env/episode_limits.hpp"
#include "obs/default_obs.hpp"
#include "reward/reward_manager.hpp"
#include "action/default_action.hpp"
#include "state/state_setter.hpp"
#include "metric/metric_manager.hpp"
#include "external/external_bot.hpp"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <random>
#include <string>
#include <cstddef>
#include <utility>

namespace buta_ppo::env {

/**
 * @brief Struct housing the result of resetting the env.
 */
struct ResetResult {
    const std::vector<float>& observations;
    const std::vector<float>& action_masks;
};

/**
 * @brief Counts of the events that make an env expensive to step, for telemetry.
 */
struct BallCostCounts {
    uint64_t touch_ticks = 0;   // Ticks in which any car touched the ball.
    uint64_t repredictions = 0; // Full ball predictions, mostly after touches.
};

/**
 * @brief Struct housing the result of a step.
 * 
 */
struct StepResult {
    // Every agent's observations.
    const std::vector<float>& observations;
    // Every agent's action_masks.
    const std::vector<float>& action_masks;
    // Every agent's accumulated rewards.
    const std::vector<float>& rewards;
    // If a terminal state was triggered.
    bool is_terminated;
    // If blue scored the goal that terminated the episode. Only meaningful when `is_terminated`.
    bool blue_scored;
    // If a truncation state was triggered.
    bool is_truncated;
    // If the truncation was from no car touching the ball for too long, rather than the match time limit.
    // Only meaningful when `is_truncated`.
    bool no_touch_truncated;
    // Ticks elapsed this step. Can be lower if a terminal/truncation state was triggered.
    int ticks_elapsed;
};

/**
 * @brief Helper struct to consolodate agent info that is needed in numerous locations.
 */
struct AgentMeta {
    uint32_t car_id;
    ffi::Team team;
};

/**
 * @brief Core wrapper managing simulation execution, buffer allocation, and state capture.
 */
class RocketSimEnv {
private:
    ffi::ArenaPtr arena_;
    ffi::ArenaState arena_state_;

    std::vector<AgentMeta> agents_;

    // The team controlled by a past policy in a PFSP match, whose rewards are left out of the telemetry.
    std::optional<ffi::Team> opponent_team_;

    std::vector<float> obs_buffer_;
    std::vector<float> action_mask_buffer_;
    std::vector<float> reward_buffer_;
    std::vector<bool> agent_x_inverted_;

    std::vector<state::StateSetterDistribution> setters_;
    std::discrete_distribution<size_t> setter_selector_;
    std::mt19937 rng_;

    int ticks_per_step_;
    size_t single_obs_size_;
    size_t action_space_size_;
    EpisodeLimits limits_;
    int ticks_since_last_touch_ = 0;
    uint64_t touch_ticks_ = 0; // Since the last `take_ball_cost_counts`.
    int match_ticks_ = 0;

    obs::DefaultObs obs_builder_;
    reward::RewardManager reward_manager_;
    action::DefaultAction action_parser_;
    metric::MetricManager metric_manager_;

    bool render_;
    bool heavy_workload_ = false; // Benchmark only, see `set_heavy_workload`.

    /**
     * @brief An external bot and the agents it drives, with scratch for batching their inference.
     */
    struct ExternalGroup {
        std::shared_ptr<external::ExternalBot> bot;
        std::vector<size_t> agents;
        std::vector<size_t> deciding; // The agents deciding this step.
        std::vector<float> obs;
        std::vector<float> decisions;
    };
    std::vector<ExternalGroup> external_groups_;
    // Per agent. Agents with a controller are driven by an external bot and ignore the policy's actions.
    std::vector<std::unique_ptr<external::ExternalController>> controllers_;

    // The kickoff countdown, 3 seconds in game.
    static constexpr int KICKOFF_COUNTDOWN_TICKS = 360;
    // The frozen ticks before the first episode, within which each external bot joins at its own random tick.
    static constexpr int JOIN_WINDOW_TICKS = 120;
    bool kickoff_pause_ = false;     // RLBot's `is_kickoff_pause`, for the external bots.
    bool external_started_ = false;  // Whether the external bots have played an episode yet.
    // Per agent: packets an external bot still waits before it joins and starts getting them. Only before the first episode.
    std::vector<int> join_ticks_;

    /**
     * @brief Plays the external bots through what they'd get before the episode's first tick in game: the frozen
     * countdown if it's a kickoff, and before the first episode, frozen ticks in which each joins at its own random tick.
     */
    void start_external_episode();

    /**
     * @brief Gives every joined external controller the current packet, decides for those that want it, and gets their
     * controls.
     *
     * @param apply_controls Whether to set the controls on the cars. Off while the cars are frozen.
     */
    void run_external_packet(bool apply_controls);

    /**
     * @brief Whether the agent is driven by an external bot that has joined the match.
     */
    [[nodiscard]] bool external_joined(size_t agent) const {
        return controllers_[agent] && (join_ticks_.empty() || join_ticks_[agent] == 0);
    }

    /**
     * @brief Gets a decision for every external controller that wants one, batched per bot.
     */
    void decide_external();

    /**
     * @brief Controls for a scripted car that chases the ball, used by the benchmark's heavy workload.
     */
    [[nodiscard]] ffi::CarControls chaser_controls(const ffi::CarState& car) const;

    /**
     * @brief Wrapper method to grab the car controls for an action given by the action parser in this env.
     * 
     * @param action_idx The action parser's discrete action lookup table idx.
     * @return ffi::CarControls The decoded car controls for this action.
     */
    ffi::CarControls decode_action(size_t action_idx);

public:
    explicit RocketSimEnv(
        const std::vector<ffi::Team>& match_layout,
        const std::vector<state::StateSetterDistribution>& setters,
        const std::vector<reward::RewardEntry>& rewards,
        const std::vector<metric::MetricEntry>& metrics,
        int ticks_per_step = 8,
        const EpisodeLimits& limits = {},
        size_t max_players_per_team = 3,
        uint32_t seed = std::random_device{}(),
        bool render = false,
        std::optional<ffi::Team> opponent_team = std::nullopt,
        const std::vector<external::ExternalSide>& external_sides = {}
    );

    /**
     * @brief Add an agent to the env.
     * 
     * @param team The team to assign them.
     * @return The agent's ID.
     */
    uint32_t add_agent(ffi::Team team);

    /**
     * @brief Reset the episode.
     * 
     * @return The result of the reset in the new arena state.
     */
    [[nodiscard]] ResetResult reset();

    /**
     * @brief Step environment and update obs and reward buffers for collection.
     * 
     * @param actions Pointer to the actions to be taken in this step for the agents.
     * @return The result of the step.
     */
    [[nodiscard]] StepResult step(const int* actions);

    /**
     * @brief Switches the env to the benchmark's heavy workload, or back.
     *
     * Every car is driven by a scripted ball-chaser instead of its actions, and the ball is repredicted at least every
     * few steps. Together they cost about as much as a bot that keeps contact with the ball, the most expensive case,
     * and behave the same on every machine. The policy still runs, so its actions are computed and then ignored.
     */
    void set_heavy_workload(bool enabled);

    /**
     * @brief Ball touch ticks and full ball repredictions since the last call, then clears them.
     */
    [[nodiscard]] BallCostCounts take_ball_cost_counts() {
        return { std::exchange(touch_ticks_, 0), obs_builder_.take_repredictions() };
    }

    /**
     * @brief Update the internal reward manager's telemetry.
     */
    void update_reward_telemetry() {
        reward_manager_.update_telemetry();
    }

    /**
     * @brief Update the internal metric manager's telemetry.
     */
    void update_metric_telemetry() {
        metric_manager_.update_telemetry();
    }

    [[nodiscard]] const std::unordered_map<std::string, reward::RewardAccumulation>& get_reward_telemetry() const { return reward_manager_.get_telemetry(); }
    [[nodiscard]] const std::unordered_map<std::string, metric::MetricAccumulation>& get_metric_telemetry() const { return metric_manager_.get_telemetry(); }
    [[nodiscard]] size_t get_obs_size() const { return single_obs_size_; }
    [[nodiscard]] size_t get_action_space_size() const { return action_space_size_; }
};

}; // namespace buta_ppo::env