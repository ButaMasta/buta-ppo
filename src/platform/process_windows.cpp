// buta-ppo/src/platform/process_windows.cpp
#include "platform/process.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace buta_ppo::platform {

namespace {

// Lock-free, so the handler may read it from whatever thread Windows runs it on.
std::atomic<std::atomic<bool>*> stop_flag{nullptr};

// Windows has no signals for these: it starts a new thread in the process and calls each registered handler on it, the
// most recently registered first, until one returns TRUE. If none does, the default handler ends the process.
BOOL WINAPI handle_console_event(DWORD event) {
    std::atomic<bool>* flag = stop_flag.load(std::memory_order_relaxed);
    switch (event) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
        // TRUE keeps the process running, so the current update can finish and save a checkpoint.
        if (flag) flag->store(true, std::memory_order_relaxed);
        return TRUE;

    case CTRL_CLOSE_EVENT:
        // The window was closed. Windows ends the process as soon as this returns, whatever it returns, or after about
        // 5 seconds otherwise. Never returning gives the main thread that time to finish the update and save; the
        // process then ends with main(), or when Windows gives up.
        if (flag) flag->store(true, std::memory_order_relaxed);
        Sleep(INFINITE);
        return TRUE;

    default:
        return FALSE;
    }
}

} // namespace

void install_stop_handler(std::atomic<bool>& flag) {
    stop_flag.store(&flag);
    SetConsoleCtrlHandler(handle_console_event, TRUE);
}

bool stdin_is_terminal() {
    // _isatty would also say yes for any character device, including NUL (`buta_ppo.exe < NUL`). Reading the console
    // mode only works on an actual console.
    DWORD mode;
    return GetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), &mode) != 0;
}

void reduce_power_throttling() {
    // Fails on Windows older than 10 1709, which doesn't throttle like this anyway, so the result is ignored.
    PROCESS_POWER_THROTTLING_STATE state{};
    state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    // Per bit, ControlMask takes the policy over from Windows and StateMask turns it on or off: the bit set in
    // ControlMask and clear in StateMask means "never throttle execution speed (EcoQoS)", whatever Windows decides.
    state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    state.StateMask = 0;
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state));
}

void disable_priority_boost() {
    // TRUE disables the boost for the process's threads, and new threads take the process's setting. If it fails, the
    // process only runs slower, so the result is ignored.
    SetProcessPriorityBoost(GetCurrentProcess(), TRUE);
}

} // namespace buta_ppo::platform
