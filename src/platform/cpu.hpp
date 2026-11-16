// buta-ppo/src/platform/cpu.hpp
#pragma once

#include <string>
#include <thread>
#include <vector>

// OS-specific code lives in src/platform/: each header is OS-neutral, and each topic has a <topic>_linux.cpp and a
// <topic>_windows.cpp, which CMake picks between. OS headers are only included by those .cpp files.
namespace buta_ppo::platform {

/**
 * @brief The logical CPUs this process may run on, in id order. Excludes CPUs left out by `taskset`.
 *
 * Reads the calling thread's allowed CPUs, so call it before that thread is pinned.
 */
std::vector<int> allowed_cpus();

/**
 * @brief Whether threads can be pinned to CPU lists on this machine. Windows can only pin a thread within one processor
 * group (at most 64 logical CPUs), so machines with several groups run unpinned. Always true on Linux.
 */
bool pinning_supported();

/**
 * @brief Checks that every CPU is one this process may run on, which excludes CPUs left out by `taskset`.
 *
 * Reads the calling thread's allowed CPUs, so call it before any thread is pinned.
 *
 * @param cpus Logical CPU ids. Empty is valid and means no pinning.
 * @param name The config field the CPUs came from, for the error message.
 * @throws std::invalid_argument If a CPU doesn't exist or this process isn't allowed to run on it, or `cpus` isn't empty
 * and `pinning_supported` is false.
 */
void validate_cpus(const std::vector<int>& cpus, const std::string& name);

/**
 * @brief Restricts a thread to a set of CPUs. The OS still moves it between those CPUs as it likes.
 *
 * On Linux, threads that the pinned thread starts afterwards inherit its CPUs. On Windows they start with the process's
 * CPUs instead.
 *
 * @param thread The thread to pin.
 * @param cpus Logical CPU ids, already checked with `validate_cpus`.
 * @return Whether the thread was pinned.
 */
bool pin_thread(std::thread& thread, const std::vector<int>& cpus);

/**
 * @brief `pin_thread` for the calling thread.
 */
bool pin_current_thread(const std::vector<int>& cpus);

/**
 * @brief Gives the calling thread back the CPUs it had at construction when it goes out of scope.
 */
class AffinityRestorer {
public:
    AffinityRestorer();
    ~AffinityRestorer();
    AffinityRestorer(const AffinityRestorer&) = delete;
    AffinityRestorer& operator=(const AffinityRestorer&) = delete;

private:
    std::vector<int> cpus_;
    bool saved_ = false;
};

/**
 * @brief A physical core and the logical CPUs (hyperthreads) on it that this process may run on.
 */
struct PhysicalCore {
    std::vector<int> cpus;
};

/**
 * @brief Groups CPUs into physical cores, fastest core first and then by lowest CPU id.
 *
 * A CPU whose topology can't be read is treated as a core of its own.
 *
 * @param cpus Logical CPU ids, e.g. from `allowed_cpus`.
 */
std::vector<PhysicalCore> physical_cores(const std::vector<int>& cpus);

} // namespace buta_ppo::platform
