// buta-ppo/src/platform/timing_linux.cpp
#include "platform/timing.hpp"

#include <thread>

namespace buta_ppo::platform {

// Linux timers already have microsecond resolution.
void precise_sleep(std::chrono::microseconds duration) {
    std::this_thread::sleep_for(duration);
}

} // namespace buta_ppo::platform
