// buta-ppo/src/main.cpp
#include "rl/ppo_runner.hpp"
#include "state/setters.hpp"
#include "reward/rewards.hpp"
#include "metric/metrics.hpp"

#include <iostream>
#include <csignal>
#include <atomic>
#include <locale>
#include <stdexcept>

namespace ffi = buta_ppo::ffi;
using namespace buta_ppo::rl;
using namespace buta_ppo::env;
using namespace buta_ppo::state;
using namespace buta_ppo::reward;
using namespace buta_ppo::metric;

std::atomic<bool> g_stop_training{false};

void handle_stop_signal(int) {
    g_stop_training.store(true, std::memory_order_relaxed);
}

int main() {
    try {
        std::cout.imbue(std::locale("en_US.utf8"));
    } catch (const std::runtime_error&) {}
    // Ctrl+C, or the stop request from a process manager (systemd, docker stop, Slurm), finishes the current update
    // and saves a checkpoint.
    std::signal(SIGINT, handle_stop_signal);
    std::signal(SIGTERM, handle_stop_signal);

    if (!ffi::init("assets/collision_meshes", true)) {
        std::cerr << "[ERROR] Failed to initialize RocketSim.\n";
        return 1;
    }
    if (!ffi::init_ball_sim("assets/collision_meshes", true)) {
        std::cerr << "[ERROR] Failed to initialize BallSim.\n";
        return 1;
    }

    RunnerConfig config;

    config.bot_name = "default";
    config.render = false;

    // Instead of training, measures thread counts, CPU pinning and env counts for this config, then prints the fastest
    // as the num_threads, num_envs, main_thread_cpus and worker_cpus lines below. See "Benchmarking Your Hardware".
    config.benchmark = false;

    config.match_distributions = {
        {1, 1, 1.0f}
    };
    config.max_players_per_team = 1;

    config.setter_distributions = {
        {create_default_kickoff_setter(), 0.30f},
        {create_random_state_setter(true, true, true), 0.35f},
        {create_random_state_setter(true, true, false), 0.35f}
    };

    // Besides a goal, episodes end when no car touches the ball for this long, or once they last this long.
    // 0 turns a limit off.
    config.no_touch_timeout_seconds = 10.0f;
    config.episode_time_limit_seconds = 150.0f;
    
    config.reward_entries = {
        {"VelocityToBall", create_velocity_to_ball_reward(), 0.1f},
        {"TouchBall", create_touch_ball_reward(), 1.0f},
        {"Goal", create_goal_reward(), 10.0f, {.opp_scale = 1.0f}}
    };

    config.metric_entries = {
        {"InAirRatio", create_in_air_ratio_metric()}
    };

    config.num_envs = 500;     // 0 picks the env count from target_steps_per_agent below.
    config.num_threads = 18;   // 0 = one per CPU in worker_cpus, or every CPU but one when unpinned.
    config.ticks_per_step = 8;

    // Optional CPU pinning by logical CPU id (see `lscpu -e`). Empty leaves the threads unpinned. Whether pinning helps
    // depends on the CPU, so let benchmark mode compare it with unpinned setups and copy its lines here.
    config.main_thread_cpus = {0};
    config.worker_cpus = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

    config.target_steps_per_update = 100'000;
    config.num_minibatches = 1;
    config.target_steps_per_agent = 100;   // Steps each agent collects per update. Sets the env count when num_envs is 0.

    // Lifetime steps between checkpoints saved during training. 0 only saves when training ends or is interrupted.
    config.steps_per_checkpoint = 100'000'000;

    // Lifetime steps between TensorBoard logs, each aggregating the updates since the last. 0 logs every update.
    config.steps_per_log = 500'000;

    config.ac_cfg.shared_layers = { 1280, 1280, 1024 };
    config.ac_cfg.actor_layers  = { 768, 512 };
    config.ac_cfg.critic_layers = { 1024, 768 };
    config.ac_cfg.use_layer_norm = true;

    // Presets: OptimizerConfig::adam(lr), ::adamw(lr, wd), ::muon(muon_lr, adamw_lr, wd).
    // Sections (shared / actor / critic) and groups within them can be adjusted afterwards, e.g. `.critic.with_lr(1e-4f)`.
    // muon lr is scaled to be comparable to adam lr.
    config.ppo_cfg.optimizer = OptimizerConfig::muon(3e-4, 3e-4, 0.01);
    config.ppo_cfg.entropy_coef = 0.035f;
    config.ppo_cfg.epochs = 1;

    // PFSP: fraction of even-team envs playing past versions, and how many versions are played each update.
    config.pfsp_cfg.envs = 0.2f;
    config.pfsp_cfg.versions = 2;

    try {
        PPORunner runner(config);
        // Trains for at most this many updates (~50M steps with this config), then saves a checkpoint and exits.
        // Rerunning resumes from that checkpoint.
        runner.run(1'000,g_stop_training, "checkpoints");
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}