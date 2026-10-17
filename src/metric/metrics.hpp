// buta-ppo/src/metric/metrics.hpp
#pragma once

#include "metric_manager.hpp"

#include <memory>

namespace buta_ppo::metric {

std::unique_ptr<MetricFunction> create_in_air_ratio_metric();

} // namespace buta_ppo::metric