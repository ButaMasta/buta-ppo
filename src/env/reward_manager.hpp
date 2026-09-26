// buta-ppo/src/env/reward_manager.hpp
#pragma once

#include "buta_ppo/ffi.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

// Forward declaration.
namespace buta_ppo::env { struct AgentMeta; }

namespace buta_ppo::env {

// Template function all other functions will extend.
class RewardFunction {
public:
    virtual ~RewardFunction() = default;

    // Called at the start of an episode.
    virtual void reset([[maybe_unused]] const ffi::ArenaState& initial_state) {}

    // Called every tick. Receives the current state and the previous state.
    virtual float get_reward(
        const AgentMeta& agent, 
        const ffi::ArenaState& current_state,
        [[maybe_unused]] const ffi::ArenaState& previous_state
    ) = 0;
};

class TouchBallReward : public RewardFunction {
public:
    float get_reward(const AgentMeta& agent, const ffi::ArenaState& state, const ffi::ArenaState& prev_state) override;
};

class GoalReward : public RewardFunction {
public:
    float get_reward(const AgentMeta& agent, const ffi::ArenaState& state, const ffi::ArenaState& prev_state) override;
};

class VelocityToBallReward : public RewardFunction {
public:
    float get_reward(const AgentMeta& agent, const ffi::ArenaState& state, const ffi::ArenaState& prev_state) override;
};

struct RewardEntry {
    std::string name;
    std::unique_ptr<RewardFunction> function;
    float weight;
    double accumulated_value;
};

class RewardManager {
private:
    std::vector<RewardEntry> rewards_;
    ffi::ArenaState previous_state_;

    std::unordered_map<std::string, double> telemetry_breakdown_;

public:
    RewardManager() = default;

    /**
     * @brief Register a reward function with a specific weight and name.
     * 
     * @param name - The visual name for the reward. (Used in metrics)
     * @param reward_func - The pointer to the function itself.
     * @param weight - The weight for the result of the reward function.
     */
    void add_reward(std::string name, std::unique_ptr<RewardFunction> reward_func, float weight = 1.0f);

    /**
     * @brief Resets all reward trackers and caches the initial state.
     * 
     * @param initial_state The initial arena state to reset to.
     */
    void reset(const ffi::ArenaState& initial_state);

    /**
     * @brief Calculates the weighted sum of all rewards for a given agent.
     * 
     * @param agent - The agent to get the rewards for.
     * @param current_state - The current arena state to utilize in reward calculation.
     * @return float - The total reward output for this agent.
     */
    float get_reward(const AgentMeta& agent, const ffi::ArenaState& current_state);

    /**
     * @brief Updates the previous state. Called after all rewards for a tick have been fetched.
     * 
     * @param arena_state - The state to set the previous state to.
     */
    void update_previous_state(const ffi::ArenaState& arena_state);

    /**
     * @brief Updates the internal telemetry map with new values.
     */
    void update_telemetry();

    /**
     * @brief Fetches the telemetry pointer.
     *
     * NOTE: Only call AFTER `update_telemetry`.
     * 
     * @return const std::unordered_map<std::string, double>& - The telemetry pointer. 
     */
    [[nodiscard]] const std::unordered_map<std::string, double>& get_telemetry() const {
        return telemetry_breakdown_;
    }
};

} // namespace buta_ppo::env