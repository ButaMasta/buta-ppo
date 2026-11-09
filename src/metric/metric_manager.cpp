// buta-ppo/src/metric/metric_manager.cpp
#include "metric_manager.hpp"
#include "env/rocketsim_env.hpp"
#include "reward/reward_manager.hpp"


namespace buta_ppo::metric {

std::optional<float> MetricFunction::reward_value(size_t input, const env::AgentMeta& agent) const {
    if (reward_values_ == nullptr || input >= reward_slots_.size() || reward_slots_[input] < 0) return std::nullopt;
    return reward_values_->values[reward_slots_[input]][agent.car_id];
}

void MetricManager::bind_rewards(reward::RewardManager& rewards) {
    for (MetricEntry& entry : metrics_) {
        entry.function->reward_slots_.clear();
        for (const std::string& name : entry.function->reward_inputs()) {
            entry.function->reward_slots_.push_back(rewards.track_tick_values(name));
        }
    }
}

void MetricManager::set_metrics(const std::vector<MetricEntry> &metrics) {
    metrics_ = metrics;
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
    for (MetricEntry& entry : metrics_) {
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