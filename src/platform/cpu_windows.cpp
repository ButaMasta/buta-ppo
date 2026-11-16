// buta-ppo/src/platform/cpu_windows.cpp
#include "platform/cpu.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace buta_ppo::platform {

// Windows splits a machine's logical CPUs into processor groups of at most 64, and an affinity is always a group number
// plus a 64-bit mask (KAFFINITY) of CPUs within that group. The config uses one flat id per logical CPU, as on Linux:
// group 0's CPUs first, then group 1's, and so on, which is also how Task Manager numbers them. Machines with up to 64
// logical CPUs have a single group, where an id is simply the bit number.
namespace {

constexpr int GROUP_BITS = 64;

// The flat id of bit 0 of `group`.
int group_offset(WORD group) {
    int offset = 0;
    for (WORD g = 0; g < group; g++) offset += static_cast<int>(GetActiveProcessorCount(g));
    return offset;
}

// Where a flat CPU id lives: its group, and its bit within that group's mask.
struct GroupBit {
    WORD group;
    int bit;
};

// Nullopt for ids no group has.
std::optional<GroupBit> locate(int cpu) {
    if (cpu < 0) return std::nullopt;
    const WORD groups = GetActiveProcessorGroupCount();
    int offset = 0;
    for (WORD g = 0; g < groups; g++) {
        const int count = static_cast<int>(GetActiveProcessorCount(g));
        if (cpu < offset + count) return GroupBit{g, cpu - offset};
        offset += count;
    }
    return std::nullopt;
}

// The flat ids of the CPUs set in one group's mask, in id order.
std::vector<int> to_cpus(WORD group, KAFFINITY mask) {
    std::vector<int> cpus;
    const int offset = group_offset(group);
    for (int b = 0; b < GROUP_BITS; b++) {
        bool bit = mask & (KAFFINITY{1} << b);
        if (bit) {
            cpus.push_back(offset + b);
        }
    }
    return cpus;
}

// The Windows affinity for a list of flat CPU ids. Nullopt if the list is empty, has an id that doesn't exist, or
// spans more than one group, since one thread's affinity can only name one group.
std::optional<GROUP_AFFINITY> to_group_affinity(const std::vector<int>& cpus) {
    if (cpus.empty()) return std::nullopt;
    GROUP_AFFINITY affinity{}; // {} also zeroes Reserved, which Windows requires.
    for (size_t i = 0; i < cpus.size(); i++) {
        const int cpu = cpus[i];
        std::optional<GroupBit> where = locate(cpu);
        if (!where) return std::nullopt;
        if (i == 0) {
            affinity.Group = where->group;
        } else if (where->group != affinity.Group) {
            return std::nullopt;
        }
        affinity.Mask |= KAFFINITY{1} << where->bit;
    }
    return affinity;
}

bool pin(HANDLE thread, const std::vector<int>& cpus) {
    const std::optional<GROUP_AFFINITY> affinity = to_group_affinity(cpus);
    return affinity && SetThreadGroupAffinity(thread, &*affinity, nullptr);
}

} // namespace

std::vector<int> allowed_cpus() {
    // With several groups, Windows 11 lets a process's threads run in all of them by default, and GetProcessAffinityMask
    // can't describe that (it returns zero masks once threads are in more than one group). Such machines run unpinned
    // (pinning_supported), so every active CPU is listed.
    if (GetActiveProcessorGroupCount() > 1) {
        std::vector<int> cpus;
        for (WORD g = 0; g < GetActiveProcessorGroupCount(); g++) {
            const int offset = group_offset(g);
            for (DWORD i = 0; i < GetActiveProcessorCount(g); i++) cpus.push_back(offset + static_cast<int>(i));
        }
        return cpus;
    }
    // One group: the process mask, which excludes CPUs left out by `start /affinity`.
    DWORD_PTR process_mask = 0;
    DWORD_PTR system_mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) || process_mask == 0) {
        throw std::runtime_error("Could not read the CPUs this process may run on.");
    }
    return to_cpus(0, process_mask);
}

bool pinning_supported() {
    return GetActiveProcessorGroupCount() == 1;
}

void validate_cpus(const std::vector<int>& cpus, const std::string& name) {
    if (cpus.empty()) return;
    if (!pinning_supported()) {
        throw std::invalid_argument(name + " is set, but CPU pinning isn't supported on Windows with more than 64 "
                                    "logical CPUs (several processor groups). Leave main_thread_cpus and worker_cpus "
                                    "empty.");
    }
    const std::vector<int> allowed = allowed_cpus();
    for (const int cpu : cpus) {
        if (std::find(allowed.begin(), allowed.end(), cpu) == allowed.end()) {
            throw std::invalid_argument(name + " has CPU " + std::to_string(cpu) + ", which this process can't run on. "
                                        "Check the CPU ids in Task Manager's logical processor view and any /affinity "
                                        "the process was started with.");
        }
    }
}

// MSVC's std::thread::native_handle() is the thread's Windows HANDLE.
bool pin_thread(std::thread& thread, const std::vector<int>& cpus) {
    return pin(thread.native_handle(), cpus);
}

// GetCurrentThread() is a pseudo-handle meaning "the calling thread", which needs no closing.
bool pin_current_thread(const std::vector<int>& cpus) {
    return pin(GetCurrentThread(), cpus);
}

// A thread's affinity is always within one group, so this saves and restores that group's CPUs.
AffinityRestorer::AffinityRestorer() {
    GROUP_AFFINITY affinity{};
    saved_ = GetThreadGroupAffinity(GetCurrentThread(), &affinity) != 0;
    if (saved_) cpus_ = to_cpus(affinity.Group, affinity.Mask);
}

AffinityRestorer::~AffinityRestorer() {
    if (saved_) {
        pin_current_thread(cpus_);
    }
}

std::vector<PhysicalCore> physical_cores(const std::vector<int>& cpus) {
    // Windows returns one variable-length record per physical core, packed back to back, each starting with its own
    // Size. Called once with no buffer for the length it needs, then again to fill it. uint64_t storage keeps the
    // records aligned.
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    std::vector<uint64_t> buffer((length + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    auto* records = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
    if (length == 0 || !GetLogicalProcessorInformationEx(RelationProcessorCore, records, &length)) {
        length = 0; // Unreadable: every CPU becomes a core of its own below.
    }

    struct Core {
        std::vector<int> cpus;
        int efficiency = 0; // EfficiencyClass: higher is a faster core (P-cores above E-cores), 0 on non-hybrid CPUs.
    };
    std::vector<Core> cores;
    std::vector<int> found;
    const auto* bytes = reinterpret_cast<const char*>(buffer.data());
    for (DWORD at = 0; at < length;) {
        const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(bytes + at);
        at += info->Size;
        if (info->Relationship != RelationProcessorCore) continue;

        Core core;
        core.efficiency = info->Processor.EfficiencyClass;
        for (WORD g = 0; g < info->Processor.GroupCount; g++) {
            const GROUP_AFFINITY& group = info->Processor.GroupMask[g];
            for (const int cpu : to_cpus(group.Group, group.Mask)) {
                if (std::find(cpus.begin(), cpus.end(), cpu) != cpus.end()) core.cpus.push_back(cpu);
            }
        }
        if (core.cpus.empty()) continue;
        std::sort(core.cpus.begin(), core.cpus.end());
        found.insert(found.end(), core.cpus.begin(), core.cpus.end());
        cores.push_back(std::move(core));
    }
    // A CPU no record names is treated as a core of its own, as on Linux.
    for (const int cpu : cpus) {
        if (std::find(found.begin(), found.end(), cpu) == found.end()) cores.push_back({{cpu}, 0});
    }

    std::sort(cores.begin(), cores.end(), [](const Core& a, const Core& b) {
        if (a.efficiency != b.efficiency) return a.efficiency > b.efficiency;
        return a.cpus.front() < b.cpus.front();
    });
    std::vector<PhysicalCore> result;
    result.reserve(cores.size());
    for (Core& core : cores) {
        result.push_back({std::move(core.cpus)});
    }
    return result;
}

} // namespace buta_ppo::platform
