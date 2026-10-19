// buta-ppo/src/util/cpu_topology.hpp
#pragma once

#include <sched.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace buta_ppo::util {

/**
 * @brief The logical CPUs this process may run on, in id order. Excludes CPUs left out by `taskset`.
 *
 * Reads the calling thread's allowed CPUs, so call it before that thread is pinned.
 */
inline std::vector<int> allowed_cpus() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        throw std::runtime_error("Could not read the CPUs this process may run on.");
    }

    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
    }
    return cpus;
}

/**
 * @brief A physical core and the logical CPUs (hyperthreads) on it that this process may run on.
 */
struct PhysicalCore {
    std::vector<int> cpus;
    long max_khz = 0; // 0 when the kernel doesn't report it.
};

/**
 * @brief Groups CPUs into physical cores, fastest core first and then by lowest CPU id.
 *
 * Reads `/sys/devices/system/cpu`. A CPU whose topology can't be read is treated as a core of its own.
 *
 * @param cpus Logical CPU ids, e.g. from `allowed_cpus`.
 */
inline std::vector<PhysicalCore> physical_cores(const std::vector<int>& cpus) {
    const auto read_long = [](const std::string& path, long fallback) {
        std::ifstream file(path);
        long value = fallback;
        if (!(file >> value)) value = fallback;
        return value;
    };

    std::map<std::pair<long, long>, PhysicalCore> cores; // Keyed by (package, core id).
    for (const int cpu : cpus) {
        const std::string dir = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
        const long package = read_long(dir + "/topology/physical_package_id", -1);
        // An unreadable core id gets a key no real core has, so the CPU stands alone.
        const long core_id = read_long(dir + "/topology/core_id", -1 - cpu);
        PhysicalCore& core = cores[{package, core_id}];
        core.cpus.push_back(cpu);
        core.max_khz = std::max(core.max_khz, read_long(dir + "/cpufreq/cpuinfo_max_freq", 0));
    }

    std::vector<PhysicalCore> result;
    for (auto& [key, core] : cores) {
        std::sort(core.cpus.begin(), core.cpus.end());
        result.push_back(std::move(core));
    }
    std::sort(result.begin(), result.end(), [](const PhysicalCore& a, const PhysicalCore& b) {
        if (a.max_khz != b.max_khz) return a.max_khz > b.max_khz;
        return a.cpus.front() < b.cpus.front();
    });
    return result;
}

} // namespace buta_ppo::util
