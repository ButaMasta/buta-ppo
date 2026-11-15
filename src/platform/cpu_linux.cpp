// buta-ppo/src/platform/cpu_linux.cpp
#include "platform/cpu.hpp"

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

namespace buta_ppo::platform {

namespace {

std::vector<int> cpus_of(const cpu_set_t& set) {
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (CPU_ISSET(cpu, &set)) cpus.push_back(cpu);
    }
    return cpus;
}

bool pin(pthread_t thread, const std::vector<int>& cpus) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const int cpu : cpus) {
        CPU_SET(cpu, &set);
    }
    return pthread_setaffinity_np(thread, sizeof(set), &set) == 0;
}

} // namespace

std::vector<int> allowed_cpus() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        throw std::runtime_error("Could not read the CPUs this process may run on.");
    }
    return cpus_of(allowed);
}

void validate_cpus(const std::vector<int>& cpus, const std::string& name) {
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

bool pin_thread(std::thread& thread, const std::vector<int>& cpus) {
    return pin(thread.native_handle(), cpus);
}

bool pin_current_thread(const std::vector<int>& cpus) {
    return pin(pthread_self(), cpus);
}

AffinityRestorer::AffinityRestorer() {
    cpu_set_t set;
    CPU_ZERO(&set);
    saved_ = pthread_getaffinity_np(pthread_self(), sizeof(set), &set) == 0;
    if (saved_) cpus_ = cpus_of(set);
}

AffinityRestorer::~AffinityRestorer() {
    if (saved_) {
        pin_current_thread(cpus_);
    }
}

std::vector<PhysicalCore> physical_cores(const std::vector<int>& cpus) {
    // Reads /sys/devices/system/cpu.
    const auto read_long = [](const std::string& path, long fallback) {
        std::ifstream file(path);
        long value = fallback;
        if (!(file >> value)) value = fallback;
        return value;
    };

    struct Core {
        std::vector<int> cpus;
        long max_khz = 0; // 0 when the kernel doesn't report it.
    };
    std::map<std::pair<long, long>, Core> cores; // Keyed by (package, core id).
    for (const int cpu : cpus) {
        const std::string dir = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
        const long package = read_long(dir + "/topology/physical_package_id", -1);
        // An unreadable core id gets a key no real core has, so the CPU stands alone.
        const long core_id = read_long(dir + "/topology/core_id", -1 - cpu);
        Core& core = cores[{package, core_id}];
        core.cpus.push_back(cpu);
        core.max_khz = std::max(core.max_khz, read_long(dir + "/cpufreq/cpuinfo_max_freq", 0));
    }

    std::vector<Core> sorted;
    for (auto& [key, core] : cores) {
        std::sort(core.cpus.begin(), core.cpus.end());
        sorted.push_back(std::move(core));
    }
    std::sort(sorted.begin(), sorted.end(), [](const Core& a, const Core& b) {
        if (a.max_khz != b.max_khz) return a.max_khz > b.max_khz;
        return a.cpus.front() < b.cpus.front();
    });

    std::vector<PhysicalCore> result;
    result.reserve(sorted.size());
    for (Core& core : sorted) {
        result.push_back({std::move(core.cpus)});
    }
    return result;
}

} // namespace buta_ppo::platform
