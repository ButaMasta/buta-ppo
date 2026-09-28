// buta-ppo/src/env/reward_manager.hpp
#pragma once

#include "buta_ppo/ffi.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

// Forward declaration.
namespace buta_ppo::env { struct AgentMeta; }

namespace buta_ppo::reward {

/**
 * @brief Abstract class for defining environment rewards.
 */
class RewardFunction {
public:
    virtual ~RewardFunction() = default;

    /**
     * @brief Resets any internal state of the reward function after an environment reset.
     * 
     * @param initial_state The starting state of the environment.
     */
    virtual void reset(const ffi::ArenaState& /*initial_state*/) {}

    // Called every tick. Receives the current state and the previous state.
    /**
     * @brief Computes the reward value for a specific agent at the current sim tick.
     * 
     * @param agent The agent recieving the reward.
     * @param current_state The current tick state.
     * @param previous_state The previous tick state.
     * @return The scalar -1.0f to 1.0f to multiply the weight of the reward by.
     */
    virtual float get_reward(
        const env::AgentMeta& agent, 
        const ffi::ArenaState& current_state,
        const ffi::ArenaState& previous_state
    ) = 0;
};

struct RewardEntry {
    std::string name;
    std::unique_ptr<RewardFunction> function;
    float weight;
    double accumulated_value;
};

/**
 * @brief Manages the set of rewards for the following actions: computing, weighting, and accumulating telemetry.
 */
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
     * @param name The visual name for the reward. (Used in metrics).
     * @param reward_func The pointer to the function itself.
     * @param weight The weight for the result of the reward function.
     */
    void add_reward(std::string name, std::unique_ptr<RewardFunction> reward_func, float weight = 1.0f);

    /**
     * @brief Resets all reward trackers with a new state at the end of an episode.
     * 
     * @param initial_state The initial arena state to reset to.
     */
    void reset(const ffi::ArenaState& initial_state);

    /**
     * @brief Calculates the weighted sum of all rewards for a given agent.
     * 
     * @param agent The agent to get the rewards for.
     * @param current_state The current arena state to utilize in reward calculation.
     * @return The total weighted reward output for this agent.
     */
    [[nodiscard]] float get_reward(const env::AgentMeta& agent, const ffi::ArenaState& current_state);

    /**
     * @brief Updates the previous state. Called after all rewards for a tick have been fetched.
     * 
     * @param arena_state The state to set the previous state to.
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

} // namespace buta_ppo::reward