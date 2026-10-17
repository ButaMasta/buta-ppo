// buta-ppo/src/metric/metric_manager.cpp
#include "metric_manager.hpp"
#include "env/rocketsim_env.hpp"


namespace buta_ppo::metric {

void MetricManager::set_metrics(const std::vector<MetricEntry> &metrics) {
    metrics_ = metrics;
    for (const MetricEntry& entry : metrics) {
        telemetry_breakdown_[entry.name] = MetricAccumulation{};
    }
}

void MetricManager::reset(const ffi::ArenaState &initial_state) {
    previous_state_ = initial_state;
}

void MetricManager::calculate_metrics(const env::AgentMeta &agent, const ffi::ArenaState &current_state, bool track_telemetry) {
    if (!track_telemetry) return;
    for (MetricEntry& entry : metrics_) {
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