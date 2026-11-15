// buta-ppo/src/platform/timing.hpp
#pragma once

#include <chrono>

namespace buta_ppo::platform {

/**
 * @brief Sleeps the calling thread for about `duration`, to well under a millisecond, e.g. to pace render to real time.
 */
void precise_sleep(std::chrono::microseconds duration);

} // namespace buta_ppo::platform
