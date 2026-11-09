// buta-ppo/src/metric/metric_manager.cpp
#include "metric_manager.hpp"
#include "env/rocketsim_env.hpp"
#include "reward/reward_manager.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace buta_ppo::metric {

std::optional<float> MetricFunction::reward_value(size_t input, const env::AgentMeta& agent) const {
    if (reward_values_ == nullptr || input >= reward_slots_.size() || reward_slots_[input] < 0) return std::nullopt;
    return reward_values_->values[reward_slots_[input]][agent.car_id];
}

bool metric_env_sampled(size_t rank, size_t group_size, float share) {
    if (group_size == 0) return false;
    const size_t k = std::clamp<size_t>(static_cast<size_t>(std::lround(share * static_cast<float>(group_size))), 1,
                                        group_size);
    // As VecEnv spreads PFSP envs: rank r is in when (r + 1) k / n passes a whole number that r k / n didn't.
    return (rank + 1) * k / group_size > rank * k / group_size;
}

std::vector<std::string> validate_metric_entries(const std::vector<MetricEntry>& metrics, float default_share,
                                                 const std::vector<std::pair<size_t, size_t>>& layouts) {
    const auto valid_share = [](float share) { return share > 0.0f && share <= 1.0f; };  // Also rejects NaN.
    if (!valid_share(default_share)) {
        throw std::invalid_argument("metric_env_share must be in (0, 1].");
    }
    std::vector<std::string> warnings;
    for (const MetricEntry& entry : metrics) {
        const std::string prefix = "Metric '" + entry.name + "': ";
        if (entry.options.env_share && !valid_share(*entry.options.env_share)) {
            throw std::invalid_argument(prefix + "env_share must be in (0, 1].");
        }
        for (const auto& [a, b] : entry.options.layouts) {
            if (a == 0 || b == 0) throw std::invalid_argument(prefix + "layouts need players on both teams.");
        }
        if (!entry.options.layouts.empty() &&
            std::none_of(layouts.begin(), layouts.end(),
                         [&](const auto& l) { return entry.options.applies_to(l.first, l.second); })) {
            warnings.push_back(prefix + "none of its layouts is played, so it never logs.");
        }
    }
    return warnings;
}

void MetricManager::select(size_t blue, size_t orange, size_t rank, size_t group_size, float default_share) {
    active_.clear();
    for (size_t i = 0; i < metrics_.size(); i++) {
        const MetricOptions& options = metrics_[i].options;
        if (!options.applies_to(blue, orange)) continue;
        if (!metric_env_sampled(rank, group_size, options.env_share.value_or(default_share))) continue;
        active_.push_back(i);
    }
}

void MetricManager::bind_rewards(reward::RewardManager& rewards) {
    for (MetricEntry& entry : metrics_) entry.function->reward_slots_.clear();
    for (const size_t i : active_) {
        MetricEntry& entry = metrics_[i];
        for (const std::string& name : entry.function->reward_inputs()) {
            entry.function->reward_slots_.push_back(rewards.track_tick_values(name));
        }
    }
}

void MetricManager::set_metrics(const std::vector<MetricEntry> &metrics) {
    metrics_ = metrics;
    active_.resize(metrics_.size());
    for (size_t i = 0; i < active_.size(); i++) active_[i] = i;
    for (const MetricEntry& entry : metrics) {
        telemetry_breakdown_[entry.name] = MetricAccumulation{};
    }
}

void MetricManager::reset(const ffi::ArenaState &initial_state) {
    previous_state_ = initial_state;
}

void MetricManager::calculate_metrics(const env::AgentMeta &agent, const ffi::ArenaState &current_state, bool track_telemetry,
                                      const reward::RewardTickValues* reward_values) {
    if (!track_telemetry) return;
    for (const size_t i : active_) {
        MetricEntry& entry = metrics_[i];
        // Passed each tick rather than kept, so a moved or copied env never leaves a metric pointing at another's.
        if (!entry.function->reward_slots_.empty()) entry.function->reward_values_ = reward_values;
        std::optional<double> metric = entry.function->get_metric(agent, current_state, previous_state_);
        if (metric.has_value()) {
            entry.accumulate.value += metric.value();
            entry.accumulate.count += 1;
        }
    }
}

void MetricManager::update_previous_state(const ffi::ArenaState &arena_state) {
    previous_state_ = arena_state;
}

void MetricManager::update_telemetry() {
    for ( MetricEntry& entry : metrics_) {
        telemetry_breakdown_[entry.name] = entry.accumulate;
        entry.accumulate.count = 0;
        entry.accumulate.value = 0.0;
    }
}

} // namespace buta_ppo::metric 