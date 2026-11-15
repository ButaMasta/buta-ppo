// buta-ppo/src/platform/process.hpp
#pragma once

#include <atomic>

namespace buta_ppo::platform {

/**
 * @brief Sets `flag` when the process is asked to stop: Ctrl+C, or the stop request from a process manager
 * (SIGTERM from systemd, docker stop, Slurm).
 *
 * @param flag Must outlive any signal, so a global or static.
 */
void install_stop_handler(std::atomic<bool>& flag);

/**
 * @brief Whether standard input is an interactive terminal, so the process may ask the user a question.
 */
bool stdin_is_terminal();

} // namespace buta_ppo::platform
