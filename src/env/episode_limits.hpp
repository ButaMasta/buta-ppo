// buta-ppo/src/env/episode_limits.hpp
#pragma once

namespace buta_ppo::env {

/**
 * @brief Physics ticks after which an episode is truncated. 0 turns a limit off.
 */
struct EpisodeLimits {
    int no_touch_ticks = 1200;     // Ticks without any car touching the ball. 10 seconds.
    int time_limit_ticks = 18'000; // Ticks since the episode started. 2.5 minutes.
};

} // namespace buta_ppo::env
