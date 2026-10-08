// buta-ppo/src/main.cpp
#include "rl/ppo_runner.hpp"
#include "state/setters.hpp"
#include "reward/rewards.hpp"

#include <iostream>
#include <csignal>
#include <atomic>
#include <locale>

namespace ffi = buta_ppo::ffi;
using namespace buta_ppo::rl;
using namespace buta_ppo::env;
using namespace buta_ppo::state;
using namespace buta_ppo::reward;

std::atomic<bool> g_stop_training{false};

void handle_sigint(int) {
    g_stop_training.store(true, std::memory_order_relaxed);
}

int main() {
    std::cout.imbue(std::locale("en_US.utf8"));
    std::signal(SIGINT, handle_sigint);

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
    config.render = true;

    config.match_distributions = {
        {1, 1, 1.0f}
    };
    config.max_players_per_team = 1;

    config.setter_distributions = {
        {create_default_kickoff_setter(), 0.30f},
        {create_random_state_setter(true, true, true), 0.35f},
        {create_random_state_setter(true, true, false), 0.35f}
    };
    
    config.reward_entries = {
        {"VelocityToBall", create_velocity_to_ball_reward(), 0.1f},
        {"TouchBall", create_touch_ball_reward(), 1.0f},
        {"Goal", create_goal_reward(), 10.0f}
    };

    config.num_envs_per_thread = 21;
    config.num_threads = 12;
    config.ticks_per_step = 8;

    config.target_steps_per_update = 50'000;
    config.num_minibatches = 2;

    config.ac_cfg.shared_layers = { 512, 512 };
    config.ac_cfg.actor_layers  = { 256, 256 };
    config.ac_cfg.critic_layers = { 512, 512 };
    config.ac_cfg.use_layer_norm = true;

    // Presets: OptimizerConfig::adam(lr), ::adamw(lr, wd), ::muon(muon_lr, adamw_lr, wd).
    // Sections (shared / actor / critic) and groups within them can be adjusted afterwards, e.g. `.critic.with_lr(1e-4f)`.
    // muon lr is scaled to be comparable to adam lr.
    config.ppo_cfg.optimizer = OptimizerConfig::muon(3e-4, 3e-4, 0.01);
    config.ppo_cfg.entropy_coef = 0.035f;
    config.ppo_cfg.epochs = 1;
    config.ppo_cfg.target_kl = 0.015f;

    PPORunner runner(config);
    runner.run(75'000, g_stop_training, "checkpoints");

    return 0;
}