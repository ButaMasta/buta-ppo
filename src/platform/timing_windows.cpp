// buta-ppo/src/platform/timing_windows.cpp
#include "platform/timing.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <chrono>
#include <thread>

namespace buta_ppo::platform {

namespace {

// A high-resolution waitable timer. Sleep() and std::this_thread::sleep_for wake on the scheduler's timer tick, 15.6 ms
// by default, so an 8.3 ms render tick would take about twice as long. This timer wakes to well under a millisecond
// without changing the tick for the rest of the system. Needs Windows 10 1803 or newer; on failure the handle is null.
class WaitableTimer {
public:
    WaitableTimer()
        : handle_(CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS)) {}

    ~WaitableTimer() {
        if (handle_) CloseHandle(handle_);
    }

    WaitableTimer(const WaitableTimer&) = delete;
    WaitableTimer& operator=(const WaitableTimer&) = delete;

    // Whether it slept. False when the timer couldn't be created or set.
    bool sleep(std::chrono::microseconds duration) {
        if (!handle_) return false;

        // In 100-nanosecond units, negative for "this long from now" (positive would be an absolute date).
        LARGE_INTEGER due_time;
        due_time.QuadPart = -static_cast<LONGLONG>(duration.count()) * 10;
        if (!SetWaitableTimer(handle_, &due_time, 0, nullptr, nullptr, FALSE)) return false;

        return WaitForSingleObject(handle_, INFINITE) == WAIT_OBJECT_0;
    }

private:
    HANDLE handle_;
};

} // namespace

void precise_sleep(std::chrono::microseconds duration) {
    if (duration <= std::chrono::microseconds::zero()) return;

    // One timer per thread, since setting a timer moves the due time of every thread waiting on it. Created on the
    // thread's first sleep and closed when the thread exits.
    thread_local WaitableTimer timer;
    if (!timer.sleep(duration)) {
        std::this_thread::sleep_for(duration);
    }
}

} // namespace buta_ppo::platform
