// buta-ppo/src/util/thread_affinity.hpp
#pragma once

#include <pthread.h>
#include <sched.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace buta_ppo::util {

/**
 * @brief Checks that every CPU is one this process may run on, which excludes CPUs left out by `taskset`.
 *
 * Reads the calling thread's allowed CPUs, so call it before any thread is pinned.
 *
 * @param cpus Logical CPU ids. Empty is valid and means no pinning.
 * @param name The config field the CPUs came from, for the error message.
 * @throws std::invalid_argument If a CPU doesn't exist or this process isn't allowed to run on it.
 */
inline void validate_cpus(const std::vector<int>& cpus, const std::string& name) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        throw std::runtime_error("Could not read the CPUs this process may run on.");
    }

    for (const int cpu : cpus) {
        if (cpu < 0 || cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed)) {
            throw std::invalid_argument(name + " has CPU " + std::to_string(cpu) + ", which this process can't run on. "
                                        "Check the CPU ids with `lscpu -e` and any `taskset` the process was started with.");
        }
    }
}

/**
 * @brief Restricts a thread to a set of CPUs. The OS still moves it between those CPUs as it likes.
 *
 * Threads that the pinned thread starts afterwards inherit its CPUs.
 *
 * @param thread The thread to pin.
 * @param cpus Logical CPU ids, already checked with `validate_cpus`.
 * @return Whether the thread was pinned.
 */
inline bool pin_thread(pthread_t thread, const std::vector<int>& cpus) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const int cpu : cpus) {
        CPU_SET(cpu, &set);
    }
    return pthread_setaffinity_np(thread, sizeof(set), &set) == 0;
}

/**
 * @brief Gives the calling thread back the CPUs it had at construction when it goes out of scope.
 */
class AffinityRestorer {
public:
    AffinityRestorer() {
        CPU_ZERO(&cpus_);
        saved_ = pthread_getaffinity_np(pthread_self(), sizeof(cpus_), &cpus_) == 0;
    }
    ~AffinityRestorer() {
        if (saved_) {
            pthread_setaffinity_np(pthread_self(), sizeof(cpus_), &cpus_);
        }
    }
    AffinityRestorer(const AffinityRestorer&) = delete;
    AffinityRestorer& operator=(const AffinityRestorer&) = delete;

private:
    cpu_set_t cpus_;
    bool saved_ = false;
};

} // namespace buta_ppo::util
