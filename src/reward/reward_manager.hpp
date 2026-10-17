// buta-ppo/src/env/reward_manager.hpp
#pragma once

#include "buta_ppo/ffi.h"

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <utility>

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

    /**
     * @brief Called once per sim tick before `get_reward` is called for any agent.
     * Use this to compute state shared across agents for the current tick.
     *
     * @param agents All agents in the environment.
     * @param current_state The current tick state.
     * @param previous_state The previous tick state.
     */
    virtual void pre_tick(
        const std::vector<env::AgentMeta>& /*agents*/,
        const ffi::ArenaState& /*current_state*/,
        const ffi::ArenaState& /*previous_state*/
    ) {}

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

    /**
     * @brief Whether this is a `TeamRewardFunction`. Don't override this, subclass `TeamRewardFunction` instead.
     */
    [[nodiscard]] virtual bool is_team_reward() const { return false; }

    [[nodiscard]] virtual std::unique_ptr<RewardFunction> clone() const = 0;
};

/**
 * @brief Abstract class for rewards that only depend on the team, such as goals or possession.
 *
 * `get_team_reward` is called once per team per sim tick, and every agent on that team receives the value.
 */
class TeamRewardFunction : public RewardFunction {
public:
    /**
     * @brief Computes the reward value for a team at the current sim tick.
     *
     * @param team The team recieving the reward.
     * @param current_state The current tick state.
     * @param previous_state The previous tick state.
     * @return The scalar -1.0f to 1.0f to multiply the weight of the reward by.
     */
    virtual float get_team_reward(
        ffi::Team team,
        const ffi::ArenaState& current_state,
        const ffi::ArenaState& previous_state
    ) = 0;

    /**
     * @brief Forwards to `get_team_reward` with the agent's team.
     */
    float get_reward(
        const env::AgentMeta& agent,
        const ffi::ArenaState& current_state,
        const ffi::ArenaState& previous_state
    ) final;

    [[nodiscard]] bool is_team_reward() const final { return true; }
};

/**
 * @brief Optional transforms applied to a reward's per-step values after they are computed for every agent.
 */
struct RewardOptions {
    // Fraction of an agent's own reward it keeps, in [0, 1]. The rest is split evenly among its teammates.
    // Has no effect for agents without teammates.
    float self_share = 1.0f;
    // How much of the opponent team's reward is subtracted from this team's, >= 0. 1 makes the reward zero-sum.
    float opp_scale = 0.0f;

    [[nodiscard]] bool is_default() const {
        return self_share == 1.0f && opp_scale == 0.0f;
    }
};

/**
 * @brief Telemetry accumulated for a reward over learner agent steps.
 */
struct RewardAccumulation {
    // Sum of the weighted values before the `RewardOptions` transforms.
    double raw = 0.0;
    // Sum of the absolute per-step values the agents received, after the transforms.
    double magnitude = 0.0;
};

struct RewardEntry {
    std::string name;
    std::unique_ptr<RewardFunction> function;
    float weight;
    RewardOptions options;
    RewardAccumulation accumulate;

    RewardEntry(std::string n, std::unique_ptr<RewardFunction> f, float w, RewardOptions o = {})
        : name(std::move(n)), function(std::move(f)), weight(w), options(o) {}

    RewardEntry(const RewardEntry& other)
        : name(other.name), function(other.function->clone()),
          weight(other.weight), options(other.options), accumulate(other.accumulate) {}

    RewardEntry& operator=(const RewardEntry& other) {
        if (this != &other) {
            name = other.name;
            function = other.function->clone();
            weight = other.weight;
            options = other.options;
            accumulate = other.accumulate;
        }
        return *this;
    }
};

/**
 * @brief Checks reward entries for invalid options, and finds settings that likely don't do what was intended.
 * Throws `std::invalid_argument` on invalid entries.
 *
 * @param rewards The reward entries to check.
 * @param largest_team The most players on one team in any match layout used.
 * @param has_empty_team Whether any match layout used has a team with no players.
 * @return Warnings for entries that are valid but likely misconfigured.
 */
[[nodiscard]] std::vector<std::string> validate_reward_entries(
    const std::vector<RewardEntry>& rewards,
    size_t largest_team,
    bool has_empty_team
);

/**
 * @brief Manages the set of rewards for the following actions: computing, weighting, transforming, and accumulating telemetry.
 *
 * Each step is `begin_step`, then `accumulate_tick` once per sim tick, then `end_step`.
 */
class RewardManager {
private:
    std::vector<RewardEntry> rewards_;
    ffi::ArenaState previous_state_;
    std::unordered_map<std::string, RewardAccumulation> telemetry_breakdown_;

    // Agent layout, bound on reset. Teams are indexed by their `ffi::Team` value.
    size_t num_agents_ = 0;
    std::vector<size_t> agent_teams_;
    std::array<size_t, 2> team_sizes_{};

    // Per-step values indexed by `[entry * num_agents_ + agent]`. Kept as members so they can be inspected after `end_step`.
    std::vector<float> step_raw_;
    std::vector<float> step_received_;

    /**
     * @brief Accumulates a team reward's weighted values for this tick, calling it once per team that has agents.
     */
    void accumulate_team_reward(TeamRewardFunction& function, float weight, float* raw, const ffi::ArenaState& current_state);

    /**
     * @brief Accumulates a per-agent reward's weighted values for this tick, calling it once per agent.
     */
    void accumulate_agent_reward(RewardFunction& function, float weight, float* raw,
                                 const std::vector<env::AgentMeta>& agents, const ffi::ArenaState& current_state);

    /**
     * @brief Applies a reward's `RewardOptions` to its step values for every agent.
     *
     * @param entry The reward the values belong to.
     * @param raw Every agent's weighted value for the step, before the options.
     * @param received Every agent's value after the options, written to.
     */
    void apply_options(const RewardEntry& entry, const float* raw, float* received) const;

public:
    RewardManager() = default;

    /**
     * @brief Set the rewards for this manager based on a vector of desired rewards.
     * Also initializes its telemetry entry.
     *
     * @param rewards The desired rewards, these are deep copied.
     */
    void set_rewards(const std::vector<RewardEntry>& rewards);

    /**
     * @brief Resets all reward trackers with a new state at the start of a new episode, and binds the agent layout.
     *
     * @param agents All agents in the environment.
     * @param initial_state The initial arena state to reset to.
     */
    void reset(const std::vector<env::AgentMeta>& agents, const ffi::ArenaState& initial_state);

    /**
     * @brief Clears the per-step values. Called before the first tick of a step.
     */
    void begin_step();

    /**
     * @brief Calls `pre_tick` on every reward, adds every agent's weighted reward for this tick to the step's values,
     * and updates the previous state.
     *
     * @param agents All agents in the environment.
     * @param current_state The current arena state.
     */
    void accumulate_tick(const std::vector<env::AgentMeta>& agents, const ffi::ArenaState& current_state);

    /**
     * @brief Applies each reward's `RewardOptions` to the step's values, adds the results to the agents' rewards,
     * and accumulates telemetry. Called after the last tick of a step.
     *
     * @param rewards Every agent's reward for the step, added to.
     * @param untracked_team The team left out of the telemetry, for agents that aren't trained on.
     */
    void end_step(std::vector<float>& rewards, std::optional<ffi::Team> untracked_team = std::nullopt);

    /**
     * @brief Updates the internal telemetry map with new values.
     */
    void update_telemetry();

    /**
     * @brief Fetches the telemetry pointer.
     *
     * NOTE: Only call AFTER `update_telemetry`.
     *
     * @return const std::unordered_map<std::string, RewardAccumulation>& - The telemetry pointer.
     */
    [[nodiscard]] const std::unordered_map<std::string, RewardAccumulation>& get_telemetry() const {
        return telemetry_breakdown_;
    }
};

} // namespace buta_ppo::reward
