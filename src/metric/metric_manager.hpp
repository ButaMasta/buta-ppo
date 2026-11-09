// buta-ppo/src/metric/metric_manager.hpp
#pragma once

#include "buta_ppo/ffi.h"

#include <optional>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <cstdint>

// Forward declarations.
namespace buta_ppo::env { struct AgentMeta; }
namespace buta_ppo::reward { struct RewardTickValues; class RewardManager; }

namespace buta_ppo::metric {

/**
 * @brief Abstract class for defining environment metrics.
 */
class MetricFunction {
    friend class MetricManager;

    // Set by the manager: this tick's reward values, and each of `reward_inputs()`'s row in them (-1 when no entry has
    // that name). Neither is episode state; the metric stays stateless.
    const reward::RewardTickValues* reward_values_ = nullptr;
    std::vector<int> reward_slots_;

protected:
    /**
     * @brief What the reward entry named `reward_inputs()[input]` returned for `agent` this tick, before its weight
     * and `RewardOptions`. Rewards run before metrics within a tick, so it's the same tick's value.
     *
     * @return The value, or nothing when no reward entry has that name (or `input` is out of range).
     */
    [[nodiscard]] std::optional<float> reward_value(size_t input, const env::AgentMeta& agent) const;

public:
    virtual ~MetricFunction() = default;

    /**
     * @brief Names of the reward entries this metric reads with `reward_value`, so it can count what a reward judges
     * (a shot, a save) without computing it again. Only these entries are recorded each tick. Empty by default.
     */
    [[nodiscard]] virtual std::vector<std::string> reward_inputs() const { return {}; }

    /**
     * @brief Computes the metric value for a specific agent at the current sim tick.
     * 
     * @param agent The agent to consider the metric for.
     * @param current_state The current tick state.
     * @param previous_state The previous tick state.
     * @return An optional that either contains nothing and conveys this tick should be ignored or a metric value.
     */
    virtual std::optional<double> get_metric(
        const env::AgentMeta& agent, 
        const ffi::ArenaState& current_state,
        const ffi::ArenaState& previous_state
    ) const = 0;


    [[nodiscard]] virtual std::unique_ptr<MetricFunction> clone() const = 0;
};

struct MetricAccumulation {
    double value = 0.0f;
    uint64_t count = 0;
};

struct MetricEntry {
    std::string name;
    std::unique_ptr<MetricFunction> function;
    MetricAccumulation accumulate;

    MetricEntry(std::string n, std::unique_ptr<MetricFunction> f)
        : name(std::move(n)), function(std::move(f)) {}
    
    MetricEntry(const MetricEntry& other)
        : name(other.name), function(other.function->clone()), 
          accumulate(other.accumulate) {}
    
    MetricEntry& operator=(const MetricEntry& other) {
        if (this != &other) {
            name = other.name;
            function = other.function->clone();
            accumulate = other.accumulate;
        }
        return *this;
    }
};

/**
 * @brief Manages the set of metrics for computing and accumulating telemetry.
 */
class MetricManager {
private:
    std::vector<MetricEntry> metrics_;
    ffi::ArenaState previous_state_;
    std::unordered_map<std::string, MetricAccumulation> telemetry_breakdown_;

public:
    MetricManager() = default;

    /**
     * @brief Set the metrics for this manager based on a vector of desired metrics. 
     * Also initializes its telemetry entry.
     * 
     * @param metrics The desired metrics, these are deep copied.
     */
    void set_metrics(const std::vector<MetricEntry>& metrics);

    /**
     * @brief Lets the metrics read the reward entries they name in `reward_inputs`: asks `rewards` to record those
     * entries each tick. Call after `set_metrics` and after the rewards are set.
     *
     * @param rewards The env's reward manager.
     */
    void bind_rewards(reward::RewardManager& rewards);

    /**
     * @brief Resets previous state with a new state at the start of a new episode.
     * 
     * @param initial_state The initial arena state to reset to.
     */
    void reset(const ffi::ArenaState& initial_state);

    /**
     * @brief Calculates the metrics for any given tick for a given agent.
     * 
     * @param agent The agent to get the metrics for.
     * @param current_state The current arena state to utilize in metric calculation.
     * @param track_telemetry Whether to add the metrics to the telemetry. False for agents that aren't trained on.
     * @param reward_values This tick's reward values from the env's reward manager, for metrics with reward inputs.
     */
    void calculate_metrics(const env::AgentMeta& agent, const ffi::ArenaState& current_state, bool track_telemetry = true,
                           const reward::RewardTickValues* reward_values = nullptr);

    /**
     * @brief Updates the previous state. Called after all metrics for a tick have been fetched.
     * 
     * @param arena_state The state to set the previous state to.
     */
    void update_previous_state(const ffi::ArenaState& arena_state);

    /**
     * @brief Updates the internal telemetry map with new values and zero accumulators.
     */
    void update_telemetry();

    /**
     * @brief Fetches the telemetry pointer.
     *
     * NOTE: Only call AFTER `update_telemetry`.
     * 
     * @return const std::unordered_map<std::string, MetricAccumulation>& - The telemetry pointer. 
     */
    [[nodiscard]] const std::unordered_map<std::string, MetricAccumulation>& get_telemetry() const {
        return telemetry_breakdown_;
    }
};

} // namespace buta_ppo::metric