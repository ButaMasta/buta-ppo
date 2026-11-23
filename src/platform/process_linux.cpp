// buta-ppo/src/platform/process_linux.cpp
#include "platform/process.hpp"

#include <unistd.h>

#include <csignal>

namespace buta_ppo::platform {

namespace {

// Lock-free, so the signal handler may read it.
std::atomic<std::atomic<bool>*> stop_flag{nullptr};

void handle_stop_signal(int) {
    if (std::atomic<bool>* flag = stop_flag.load(std::memory_order_relaxed)) {
        flag->store(true, std::memory_order_relaxed);
    }
}

} // namespace

void install_stop_handler(std::atomic<bool>& flag) {
    stop_flag.store(&flag);
    std::signal(SIGINT, handle_stop_signal);
    std::signal(SIGTERM, handle_stop_signal);
}

bool stdin_is_terminal() {
    return isatty(STDIN_FILENO);
}

void reduce_power_throttling() {}

void disable_priority_boost() {}

} // namespace buta_ppo::platform
