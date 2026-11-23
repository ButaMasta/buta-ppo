// buta-ppo/src/platform/process.hpp
#pragma once

#include <atomic>

namespace buta_ppo::platform {

/**
 * @brief Sets `flag` when the process is asked to stop: Ctrl+C, the stop request from a process manager on Linux
 * (SIGTERM from systemd, docker stop, Slurm), or Ctrl+Break or closing the console window on Windows. A closed window
 * gives the process about 5 seconds before Windows ends it, which may or may not be enough to save a checkpoint.
 *
 * @param flag Must outlive any signal, so a global or static.
 */
void install_stop_handler(std::atomic<bool>& flag);

/**
 * @brief Whether standard input is an interactive terminal, so the process may ask the user a question.
 */
bool stdin_is_terminal();

/**
 * @brief Asks the OS not to slow the process down to save power while its window is in the background.
 *
 * Windows 11 runs a process whose console isn't focused at lower clocks or on efficiency cores (EcoQoS), which would
 * slow training whenever another window is in front. Does nothing on Linux, which has no such policy.
 */
void reduce_power_throttling();

/**
 * @brief Stops the OS from briefly raising a thread's priority when it wakes from a wait, for every thread of the
 * process, including ones started later.
 *
 * Windows boosts a thread that wakes up. When the main thread wakes the workers to step the physics, the boosted workers
 * can take the CPU it needs to go on with uploads and inference, so it often resumed only 0.4–0.7 ms later. Turning the
 * boost off made training on Windows about 17% faster. Does nothing on Linux, whose scheduler has no such boost.
 */
void disable_priority_boost();

} // namespace buta_ppo::platform
