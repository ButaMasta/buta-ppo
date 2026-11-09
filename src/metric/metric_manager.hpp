// buta-ppo/src/metric/metric_manager.hpp
#pragma once

#include "buta_ppo/ffi.h"

#include <cstddef>
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

/**
 * @brief Where a metric entry is computed. Metrics run every tick for every measured agent, so leaving them out of envs
 * where they can't apply, or measuring them in a share of the envs, saves their cost there.
 */
struct MetricOptions {
    // Team sizes {blue, orange} of the match layouts the entry is computed in, each matching either way round ({1, 2}
    // covers 1v2 and 2v1). Empty: every layout. Other envs never call it, which is how to log a metric per layout.
    std::vector<std::pair<size_t, size_t>> layouts = {};
    // Share of each layout's envs that compute the entry, in (0, 1], spread evenly over its self-play envs and over its
    // PFSP envs. Unset: the config's `metric_env_share` (1 by default, every env).
    std::optional<float> env_share = std::nullopt;

    /**
     * @brief Whether the entry is computed in a `blue` v `orange` layout.
     */
    [[nodiscard]] bool applies_to(size_t blue, size_t orange) const {
        if (layouts.empty()) return true;
        for (const auto& [a, b] : layouts) {
            if ((a == blue && b == orange) || (a == orange && b == blue)) return true;
        }
        return false;
    }
};

struct MetricEntry {
    std::string name;
    std::unique_ptr<MetricFunction> function;
    MetricOptions options;
    MetricAccumulation accumulate;

    MetricEntry(std::string n, std::unique_ptr<MetricFunction> f, MetricOptions o = {})
        : name(std::move(n)), function(std::move(f)), options(std::move(o)) {}
    
    MetricEntry(const MetricEntry& other)
        : name(other.name), function(other.function->clone()), options(other.options),
          accumulate(other.accumulate) {}
    
    MetricEntry& operator=(const MetricEntry& other) {
        if (this != &other) {
            name = other.name;
            function = other.function->clone();
            options = other.options;
            accumulate = other.accumulate;
        }
        return *this;
    }
};

/**
 * @brief Whether the env at `rank` among the `group_size` envs of its layout computes an entry measured in a `share`
 * of them: max(1, round(share x group_size)) envs, spread evenly over the ranks.
 */
[[nodiscard]] bool metric_env_sampled(size_t rank, size_t group_size, float share);

/**
 * @brief Checks metric entries' options. Throws `std::invalid_argument` on a share outside (0, 1] (`default_share`
 * or an entry's) or a layout with an empty team.
 *
 * @param metrics The metric entries to check.
 * @param default_share The config's `metric_env_share`.
 * @param layouts Team sizes {blue, orange} of the match layouts that get envs.
 * @return Warnings for entries that are valid but likely misconfigured, such as layouts that no env plays.
 */
[[nodiscard]] std::vector<std::string> validate_metric_entries(
    const std::vector<MetricEntry>& metrics,
    float default_share,
    const std::vector<std::pair<size_t, size_t>>& layouts
);

/**
 * @brief Manages the set of metrics for computing and accumulating telemetry.
 */
class MetricManager {
private:
    std::vector<MetricEntry> metrics_;
    // Indices of the entries this env computes (`select`); all of them until then.
    std::vector<size_t> active_;
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
     * @brief Picks the entries this env computes from their `MetricOptions`: those whose layouts include this env's
     * and that sample this env (`metric_env_sampled`). The others are never called here; they keep their telemetry
     * entry with no measured ticks. Call `bind_rewards` again afterwards.
     *
     * @param blue Players on blue in this env.
     * @param orange Players on orange in this env.
     * @param rank This env's index among the envs of its layout.
     * @param group_size The number of envs of its layout.
     * @param default_share The share for entries without their own `env_share`.
     */
    void select(size_t blue, size_t orange, size_t rank, size_t group_size, float default_share);

    /**
     * @brief How many entries this env computes.
     */
    [[nodiscard]] size_t active_count() const { return active_.size(); }

    /**
     * @brief Lets the computed metrics read the reward entries they name in `reward_inputs`: asks `rewards` to record
     * those entries each tick. Call after `set_metrics` (and `select`) and after the rewards are set.
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