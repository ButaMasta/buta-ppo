// buta-ppo/src/metric/in_air_ratio.cpp
#include "metric_manager.hpp"
#include "metrics.hpp"
#include "env/rocketsim_env.hpp"

#include <memory>
#include <optional>

namespace buta_ppo::metric {

class InAirRatioMetric : public MetricFunction {
public:
    std::optional<double> get_metric(
        const env::AgentMeta& agent, 
        const ffi::ArenaState& current_state,
        const ffi::ArenaState& /* previous_state */
    ) const override {
        return current_state.cars[agent.car_id].is_on_ground ? 0.0f : 1.0f;
    }

    [[nodiscard]] std::unique_ptr<MetricFunction> clone() const override {
        return std::make_unique<InAirRatioMetric>(*this);
    }
};

std::unique_ptr<MetricFunction> create_in_air_ratio_metric() {
    return std::make_unique<InAirRatioMetric>();
}

} // namespace buta_ppo::metric