// buta-ppo/src/rl/ppo_benchmark.cpp
// The benchmark mode of `PPORunner`, see `RunnerConfig::benchmark`.
#include "ppo_runner.hpp"
#include "env/vec_env.hpp"
#include "util/cpu_topology.hpp"
#include "util/thread_affinity.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace buta_ppo::rl {

namespace {

// Updates run before each sample is measured, so the switch from the previous workload or config settles.
constexpr int WARMUP_UPDATES = 3;
// Updates measured per sample. Single updates vary by about 6%, so the median over both rounds is good to about 1%.
constexpr int MEASURED_UPDATES = 30;
// Every config is measured once per round, in reverse order in the second, so slow drift (heat, clocks) cancels out.
constexpr int ROUNDS = 2;

// Env counts measured for the fastest CPU setup, as multiples of the target env count. The close ones catch the steps
// in inference cost as the batch crosses GPU tile sizes. Only env counts with steps per agent within RECOMMEND_RANGE of
// the target can be recommended, and only when their Worst beats the target's by LEAVE_TARGET_MARGIN, well above the
// ~1% measurement noise, since steps per agent is a learning choice. The others show what moving further would gain.
constexpr double ENV_FACTORS[] = {0.5, 0.75, 0.95, 1.05, 1.33, 2.0};
constexpr double RECOMMEND_RANGE = 1.4;
constexpr double LEAVE_TARGET_MARGIN = 0.05;
// An unpinned setup is recommended over a faster pinned one within this much Worst. Pinned CPU ids don't carry over to
// another machine or a `taskset`, and a difference this small is close to the measurement noise.
constexpr double PREFER_UNPINNED_MARGIN = 0.02;

// Assumed speed and time to build a config's runner, for the time estimate shown before starting.
constexpr double ESTIMATE_SPS = 500'000.0;
constexpr double ESTIMATE_SECONDS_PER_BUILD = 3.0;

struct CpuSetup {
    std::string name;
    size_t threads = 0;
    std::vector<int> main_cpus;
    std::vector<int> worker_cpus;
};

/**
 * @brief One measured config and its results.
 */
struct Candidate {
    CpuSetup setup;
    size_t envs = 0;
    size_t steps_per_agent = 0;
    int64_t steps_per_update = 0;
    // Indexed light, heavy. Every measured update of every round.
    std::vector<double> update_seconds[2];
    std::vector<double> rollout_seconds[2];
    std::vector<double> physics_wait[2];
    // Why the config couldn't be built, e.g. too few PFSP envs per group at this env count. Empty when it could.
    std::string skip_reason;

    [[nodiscard]] bool measured() const { return !update_seconds[0].empty() && !update_seconds[1].empty(); }
};

double median(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    return *mid;
}

std::string cpu_list(const std::vector<int>& cpus) {
    std::ostringstream out;
    out << "{";
    for (size_t i = 0; i < cpus.size(); i++) {
        out << (i == 0 ? "" : ", ") << cpus[i];
    }
    out << "}";
    return out.str();
}

// Compact form for the table, e.g. "0, 2-19".
std::string cpu_ranges(const std::vector<int>& cpus) {
    std::ostringstream out;
    for (size_t i = 0; i < cpus.size();) {
        size_t j = i;
        while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1) j++;
        out << (i == 0 ? "" : ", ") << cpus[i];
        if (j > i) out << "-" << cpus[j];
        i = j + 1;
    }
    return out.str();
}

bool is_pinned(const CpuSetup& setup) {
    return !setup.main_cpus.empty() || !setup.worker_cpus.empty();
}

bool same_setup(const CpuSetup& a, const CpuSetup& b) {
    return a.threads == b.threads && a.main_cpus == b.main_cpus && a.worker_cpus == b.worker_cpus;
}

/**
 * @brief The CPU setups to compare: the current config, a few unpinned thread counts, and the main thread pinned.
 */
std::vector<CpuSetup> make_cpu_setups(const RunnerConfig& config, const std::vector<int>& allowed) {
    std::vector<CpuSetup> setups;
    const auto add = [&](CpuSetup setup) {
        if (setup.threads == 0) return;
        for (const CpuSetup& existing : setups) {
            if (same_setup(existing, setup)) return;
        }
        setups.push_back(std::move(setup));
    };

    const size_t cpus = allowed.size();
    const std::vector<util::PhysicalCore> cores = util::physical_cores(allowed);

    add({"Current config", config.num_threads, config.main_thread_cpus, config.worker_cpus});
    add({"Unpinned, a worker per CPU", cpus, {}, {}});
    if (cpus > 1) add({"Unpinned, 1 CPU spare", cpus - 1, {}, {}});
    if (cpus > 2) add({"Unpinned, 2 CPUs spare", cpus - 2, {}, {}});
    if (cores.size() < cpus) add({"Unpinned, a worker per core", cores.size(), {}, {}});

    // The main thread on the fastest core: alone, so no worker shares the core with it, or with workers on its
    // other hyperthreads. The workers share every other CPU.
    if (cpus >= 3) {
        const util::PhysicalCore& fastest = cores.front();
        const int main_cpu = fastest.cpus.front();

        std::vector<int> other_cores;
        std::vector<int> other_cpus;
        for (const int cpu : allowed) {
            if (std::find(fastest.cpus.begin(), fastest.cpus.end(), cpu) == fastest.cpus.end()) other_cores.push_back(cpu);
            if (cpu != main_cpu) other_cpus.push_back(cpu);
        }
        add({"Main pinned alone on a core", other_cores.size(), {main_cpu}, other_cores});
        add({"Main pinned, workers on the rest", other_cpus.size(), {main_cpu}, other_cpus});
    }
    return setups;
}

double total_sps(const Candidate& c, int workload) {
    return static_cast<double>(c.steps_per_update) / median(c.update_seconds[workload]);
}

/**
 * @brief Each candidate's worst share of the best total SPS over the two workloads.
 */
std::vector<double> worst_shares(const std::vector<Candidate>& candidates) {
    double best[2] = {0.0, 0.0};
    for (const Candidate& c : candidates) {
        for (int w = 0; w < 2; w++) best[w] = std::max(best[w], total_sps(c, w));
    }
    std::vector<double> shares;
    for (const Candidate& c : candidates) {
        shares.push_back(std::min(total_sps(c, 0) / best[0], total_sps(c, 1) / best[1]));
    }
    return shares;
}

size_t best_index(const std::vector<Candidate>& candidates, const std::vector<bool>& eligible) {
    const std::vector<double> shares = worst_shares(candidates);
    size_t best = 0;
    double best_share = -1.0;
    for (size_t i = 0; i < candidates.size(); i++) {
        if (eligible[i] && shares[i] > best_share) {
            best_share = shares[i];
            best = i;
        }
    }
    return best;
}

void print_table(const std::vector<Candidate>& candidates, size_t marked) {
    const std::vector<double> shares = worst_shares(candidates);
    std::cout << "    " << std::left << std::setw(42) << "CPU setup" << std::right
              << std::setw(8) << "Threads" << std::setw(6) << "Envs" << std::setw(13) << "Steps/agent"
              << std::setw(12) << "Light SPS" << std::setw(12) << "Heavy SPS" << std::setw(12) << "Heavy wait"
              << std::setw(8) << "Worst" << "\n";
    for (size_t i = 0; i < candidates.size(); i++) {
        const Candidate& c = candidates[i];
        std::string name = c.setup.name;
        if (!c.setup.main_cpus.empty()) name += " (CPU " + cpu_ranges(c.setup.main_cpus) + ")";
        std::cout << (i == marked ? "  > " : "    ") << std::left << std::setw(42) << name << std::right
                  << std::setw(8) << c.setup.threads << std::setw(6) << c.envs << std::setw(13) << c.steps_per_agent
                  << std::setw(12) << static_cast<int64_t>(total_sps(c, 0))
                  << std::setw(12) << static_cast<int64_t>(total_sps(c, 1))
                  << std::setw(11) << std::fixed << std::setprecision(0) << 100.0 * median(c.physics_wait[1]) << "%"
                  << std::setw(7) << std::setprecision(1) << 100.0 * shares[i] << "%" << std::defaultfloat << "\n";
    }
}

} // namespace

PPORunner::UpdateSample PPORunner::measure_updates(bool heavy, int warmup, int measured) {
    if (rollout_groups_.empty()) {
        prepare_rollouts();
        // The opponents play the learner, as before PFSP starts.
        for (size_t slot = 0; slot < opponent_slots_.size(); slot++) {
            load_opponent_slot(slot, actor_critic_->parameters());
        }
    }
    vec_env_->set_heavy_workload(heavy);

    UpdateSample sample;
    for (int update = 0; update < warmup + measured; update++) {
        const auto start = std::chrono::steady_clock::now();
        const double wait_seconds = collect_rollout();
        // Only cleared, so the counts don't build up over the benchmark.
        (void)vec_env_->collect_episode_counts();
        (void)vec_env_->collect_pfsp_outcomes();
        const auto rollout_end = std::chrono::steady_clock::now();

        compute_advantages();
        (void)trainer_->train_step(*buffer_);
        const auto end = std::chrono::steady_clock::now();

        const double rollout_seconds = std::chrono::duration<double>(rollout_end - start).count();
        const double update_seconds = std::chrono::duration<double>(end - start).count();
        if (update >= warmup) {
            sample.update_seconds.push_back(update_seconds);
            sample.rollout_seconds.push_back(rollout_seconds);
            sample.physics_wait.push_back(wait_seconds / rollout_seconds);
        }
    }
    return sample;
}

void PPORunner::run_benchmark(const std::atomic<bool>& stop_flag) {
    // Read before anything is pinned. Unpinned configs run on all of these CPUs.
    const std::vector<int> allowed = util::allowed_cpus();
    const auto unpin_main = [&] { (void)util::pin_thread(pthread_self(), allowed); };
    const auto pin_main = [&](const CpuSetup& setup) {
        if (setup.main_cpus.empty()) {
            unpin_main();
        } else if (!util::pin_thread(pthread_self(), setup.main_cpus)) {
            std::cerr << "[WARN] Could not pin the main thread to " << cpu_list(setup.main_cpus) << ".\n";
        }
    };

    const std::vector<CpuSetup> setups = make_cpu_setups(config_, allowed);
    const std::vector<util::PhysicalCore> cores = util::physical_cores(allowed);
    // Resolved by the constructor from num_envs or target_steps_per_agent.
    const size_t target_envs = config_.num_envs;

    const auto make_candidate = [&](const CpuSetup& setup, double env_factor) {
        Candidate c;
        c.setup = setup;
        c.envs = std::max<size_t>(2, static_cast<size_t>(std::lround(static_cast<double>(target_envs) * env_factor)));
        return c;
    };
    // The main thread is unpinned first, since the workers of an unpinned config inherit its CPUs.
    const auto make_runner = [&](const Candidate& c) {
        RunnerConfig cfg = config_;
        cfg.num_threads = c.setup.threads;
        cfg.num_envs = c.envs;
        cfg.main_thread_cpus = c.setup.main_cpus;
        cfg.worker_cpus = c.setup.worker_cpus;
        unpin_main();
        return std::unique_ptr<PPORunner>(new PPORunner(cfg, true));
    };

    const size_t num_configs = setups.size() + std::size(ENV_FACTORS);
    const double updates = static_cast<double>(num_configs * 2 * ROUNDS * (WARMUP_UPDATES + MEASURED_UPDATES));
    const double estimate_seconds = updates * static_cast<double>(config_.target_steps_per_update) / ESTIMATE_SPS
                                  + static_cast<double>(num_configs * ROUNDS) * ESTIMATE_SECONDS_PER_BUILD;

    std::cout << "\nBenchmark\n"
              << " | CPUs:     " << allowed.size() << " logical on " << cores.size() << " cores ("
              << cpu_ranges(allowed) << ")\n"
              << " | Envs:     aiming at " << target_envs << " envs, about "
              << config_.target_steps_per_update / std::max<size_t>(1, learner_agents_for(target_envs, config_))
              << " steps per agent at " << config_.target_steps_per_update << " steps per update\n"
              << " | Measures " << setups.size() << " CPU setups, then " << std::size(ENV_FACTORS)
              << " more env counts for the fastest, each under a light and a heavy workload.\n"
              << " | Takes about " << std::max(1, static_cast<int>(std::lround(estimate_seconds / 60.0)))
              << " min. Ctrl+C stops early and prints what was measured.\n"
              << " | Run it the way you train: keep open what usually runs alongside.\n";

    if (isatty(STDIN_FILENO)) {
        std::cout << "Run the benchmark? [y/N] " << std::flush;
        std::string answer;
        std::getline(std::cin, answer);
        if (stop_flag || (answer != "y" && answer != "Y" && answer != "yes")) {
            std::cout << "Benchmark cancelled.\n";
            return;
        }
    }

    // Measures every candidate under both workloads, with training updates like `run_training`. A runner holds a
    // trainer, so only one is built at a time, once per round. The second round runs in reverse order.
    const auto measure = [&](std::vector<Candidate>& candidates) {
        for (int round = 0; round < ROUNDS; round++) {
            for (size_t k = 0; k < candidates.size(); k++) {
                const size_t i = round % 2 == 0 ? k : candidates.size() - 1 - k;
                Candidate& c = candidates[i];
                if (stop_flag) return false;
                if (!c.skip_reason.empty()) continue;

                // A config this env count or setup can't run, like too few PFSP envs per group, is skipped and listed.
                std::unique_ptr<PPORunner> runner;
                try {
                    runner = make_runner(c);
                } catch (const std::invalid_argument& e) {
                    c.skip_reason = e.what();
                    continue;
                }
                c.steps_per_agent = runner->buffer_size_;
                c.steps_per_update = runner->total_steps_per_update_;
                pin_main(c.setup);
                for (int n = 0; n < 2; n++) {
                    if (stop_flag) break;
                    const int workload = (round + n) % 2;
                    std::cout << "\r | Round " << round + 1 << "/" << ROUNDS << ", config " << k + 1 << "/"
                              << candidates.size() << (workload ? ", heavy" : ", light") << "    " << std::flush;
                    const UpdateSample sample = runner->measure_updates(workload == 1, WARMUP_UPDATES, MEASURED_UPDATES);
                    c.update_seconds[workload].insert(c.update_seconds[workload].end(), sample.update_seconds.begin(), sample.update_seconds.end());
                    c.rollout_seconds[workload].insert(c.rollout_seconds[workload].end(), sample.rollout_seconds.begin(), sample.rollout_seconds.end());
                    c.physics_wait[workload].insert(c.physics_wait[workload].end(), sample.physics_wait.begin(), sample.physics_wait.end());
                }
                runner.reset();
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }
        unpin_main();
        std::cout << "\r" << std::string(48, ' ') << "\r";
        return !stop_flag;
    };
    const auto print_skipped = [](const std::vector<Candidate>& candidates) {
        for (const Candidate& c : candidates) {
            if (c.skip_reason.empty()) continue;
            std::cout << " | Skipped " << c.setup.name << " at " << c.envs << " envs: " << c.skip_reason << "\n";
        }
    };
    const auto measured_only = [](const std::vector<Candidate>& candidates) {
        std::vector<Candidate> measured;
        std::copy_if(candidates.begin(), candidates.end(), std::back_inserter(measured),
                     [](const Candidate& c) { return c.measured(); });
        return measured;
    };

    // Phase 1: the CPU setups at the same env count.
    std::cout << "\nComparing CPU setups...\n";
    std::vector<Candidate> cpu_candidates;
    for (const CpuSetup& setup : setups) {
        cpu_candidates.push_back(make_candidate(setup, 1.0));
    }
    const bool finished_cpu = measure(cpu_candidates);
    const std::vector<Candidate> measured_cpu = measured_only(cpu_candidates);
    if (measured_cpu.empty()) {
        const auto skipped = std::find_if(cpu_candidates.begin(), cpu_candidates.end(),
                                          [](const Candidate& c) { return !c.skip_reason.empty(); });
        if (skipped != cpu_candidates.end()) {
            std::cout << "No config could be built at " << target_envs << " envs: " << skipped->skip_reason << "\n";
        } else {
            std::cout << "Benchmark stopped before any config was measured.\n";
        }
        return;
    }

    // Training takes about the same time in every config, so the first config's light updates show it.
    {
        const Candidate& first = measured_cpu.front();
        std::vector<double> train_seconds;
        for (size_t u = 0; u < first.update_seconds[0].size(); u++) {
            train_seconds.push_back(first.update_seconds[0][u] - first.rollout_seconds[0][u]);
        }
        std::cout << " | Train:    " << std::fixed << std::setprecision(4) << median(train_seconds) << "s per update of "
                  << first.steps_per_update << " steps in " << config_.num_minibatches
                  << (config_.num_minibatches == 1 ? " minibatch\n" : " minibatches\n") << std::defaultfloat;
    }

    // Total SPS includes training. Worst is the lower of the light and heavy SPS, each as a share of the best in its
    // column, so a setup that's fast in both workloads scores highest.
    size_t best_cpu = best_index(measured_cpu, std::vector<bool>(measured_cpu.size(), true));
    bool preferred_unpinned = false;
    if (is_pinned(measured_cpu[best_cpu].setup)) {
        std::vector<bool> unpinned;
        for (const Candidate& c : measured_cpu) unpinned.push_back(!is_pinned(c.setup));
        if (std::any_of(unpinned.begin(), unpinned.end(), [](bool u) { return u; })) {
            const size_t best_unpinned = best_index(measured_cpu, unpinned);
            const std::vector<double> shares = worst_shares(measured_cpu);
            if (shares[best_cpu] - shares[best_unpinned] <= PREFER_UNPINNED_MARGIN) {
                best_cpu = best_unpinned;
                preferred_unpinned = true;
            }
        }
    }
    std::cout << "CPU setups (total SPS, > chosen):\n";
    print_table(measured_cpu, best_cpu);
    print_skipped(cpu_candidates);
    if (preferred_unpinned) {
        std::cout << "Chose the fastest unpinned setup, within " << std::lround(100.0 * PREFER_UNPINNED_MARGIN)
                  << " points of a pinned one. Pinned CPU ids don't carry over to other machines.\n";
    }

    // Phase 2: env counts for the fastest setup.
    std::vector<Candidate> env_candidates = {measured_cpu[best_cpu]};
    std::vector<Candidate> env_attempts; // Every env count tried, for listing the ones that couldn't be built.
    if (finished_cpu) {
        std::cout << "\nComparing env counts...\n";
        std::vector<Candidate> to_measure;
        for (const double factor : ENV_FACTORS) {
            Candidate c = make_candidate(measured_cpu[best_cpu].setup, factor);
            const bool duplicate = c.envs == measured_cpu[best_cpu].envs ||
                std::any_of(to_measure.begin(), to_measure.end(), [&](const Candidate& o) { return o.envs == c.envs; });
            if (!duplicate) to_measure.push_back(c);
        }
        (void)measure(to_measure);
        for (const Candidate& c : measured_only(to_measure)) {
            env_candidates.push_back(c);
        }
        env_attempts = to_measure;
        std::sort(env_candidates.begin(), env_candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.envs < b.envs; });
    }

    // The target env count is recommended, unless one with steps per agent near it is clearly faster.
    const Candidate& target = measured_cpu[best_cpu];
    const auto target_index = static_cast<size_t>(std::distance(env_candidates.begin(),
        std::find_if(env_candidates.begin(), env_candidates.end(), [&](const Candidate& c) { return c.envs == target.envs; })));
    const auto wanted_steps = static_cast<double>(target.steps_per_agent);
    std::vector<bool> eligible;
    for (const Candidate& c : env_candidates) {
        const double ratio = static_cast<double>(c.steps_per_agent) / wanted_steps;
        eligible.push_back(ratio <= RECOMMEND_RANGE && ratio >= 1.0 / RECOMMEND_RANGE);
    }
    const std::vector<double> env_shares = worst_shares(env_candidates);
    const size_t fastest_near = best_index(env_candidates, eligible);
    const bool leave_target = env_shares[fastest_near] - env_shares[target_index] >= LEAVE_TARGET_MARGIN;
    const size_t best_env = leave_target ? fastest_near : target_index;
    if (env_candidates.size() > 1) {
        std::cout << "Env counts for that setup (> recommended):\n";
        print_table(env_candidates, best_env);
        const Candidate& near = env_candidates[fastest_near];
        if (leave_target) {
            std::cout << "Recommended " << near.envs << " envs (" << near.steps_per_agent << " steps per agent) over the target's "
                      << target.steps_per_agent << ", since it scored " << std::fixed << std::setprecision(1)
                      << 100.0 * (env_shares[fastest_near] - env_shares[target_index]) << std::defaultfloat
                      << " points higher. Set num_envs to keep the target instead.\n";
        } else if (fastest_near != target_index && env_shares[fastest_near] - env_shares[target_index] >= 0.01) {
            std::cout << "Kept the target's " << target.steps_per_agent << " steps per agent. " << near.envs << " envs ("
                      << near.steps_per_agent << " steps per agent) scored " << std::fixed << std::setprecision(1)
                      << 100.0 * (env_shares[fastest_near] - env_shares[target_index]) << std::defaultfloat
                      << " points higher, under the " << std::lround(100.0 * LEAVE_TARGET_MARGIN) << " needed to move.\n";
        }
    }
    print_skipped(env_attempts);

    // This runner allocates nothing itself, so the peak since startup is the benchmark runners' peak.
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(c10::cuda::current_device());
    const auto peak_bytes = static_cast<double>(stats.reserved_bytes[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)].peak);
    const auto total_bytes = static_cast<double>(at::cuda::getCurrentDeviceProperties()->totalGlobalMem);

    const Candidate& pick = env_candidates[best_env];
    std::cout << "\nLight: a fresh policy, mostly in the air. Heavy: scripted cars chasing the ball, about as costly as\n"
              << "a bot that keeps touching it. Heavy wait is the share of the rollout spent waiting on physics.\n"
              << "Peak VRAM: " << std::fixed << std::setprecision(1) << peak_bytes / 1e9 << " GB of " << total_bytes / 1e9
              << " GB" << std::defaultfloat << "\n"
              << "\nRecommended config:\n";

    // Values the automatic settings give are printed as 0, so they keep fitting if the machine or the steps per update
    // change. The thread count is automatic like in the constructor, and the env count when target_steps_per_agent
    // picks it.
    const size_t auto_threads = pick.setup.worker_cpus.empty() ? std::max<size_t>(1, allowed.size() - 1)
                                                               : pick.setup.worker_cpus.size();
    size_t auto_envs = 0;
    if (config_.target_steps_per_agent > 0) {
        RunnerConfig from_target = config_;
        from_target.num_envs = 0;
        auto_envs = from_target.total_envs();
    }
    // Each line is one value to paste, with a note aligned after it.
    const auto print_line = [](const std::string& code, const std::string& note) {
        constexpr size_t NOTE_COLUMN = 40;
        std::cout << "    " << code;
        if (!note.empty()) std::cout << std::string(code.size() < NOTE_COLUMN ? NOTE_COLUMN - code.size() : 1, ' ') << "// " << note;
        std::cout << "\n";
    };
    if (pick.setup.threads == auto_threads) {
        print_line("config.num_threads = 0;", std::to_string(pick.setup.threads) + " workers on this machine.");
    } else {
        print_line("config.num_threads = " + std::to_string(pick.setup.threads) + ";", "");
    }
    if (pick.envs == auto_envs) {
        print_line("config.num_envs = 0;", std::to_string(pick.envs) + " envs from target_steps_per_agent = "
                   + std::to_string(config_.target_steps_per_agent) + ".");
    } else {
        print_line("config.num_envs = " + std::to_string(pick.envs) + ";", "");
    }
    print_line("config.main_thread_cpus = " + cpu_list(pick.setup.main_cpus) + ";", pick.setup.main_cpus.empty() ? "Unpinned." : "");
    print_line("config.worker_cpus = " + cpu_list(pick.setup.worker_cpus) + ";", pick.setup.worker_cpus.empty() ? "Unpinned." : "");
}

} // namespace buta_ppo::rl
