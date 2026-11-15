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

} // namespace buta_ppo::platform
