// buta-ppo/src/main.cpp
#include "rl/ppo_runner.hpp"
#include "env/state_setter.hpp"

#include <iostream>
#include <csignal>
#include <atomic>
#include <locale>

namespace ffi = buta_ppo::ffi;
using namespace buta_ppo::rl;
using namespace buta_ppo::env;

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
    config.render = false;

    config.match_distributions = {
        {1, 1, 1.0f}
    };

    config.setter_distributions = {
        {DefaultKickoffSetter(), 0.40f},
        {RandomStateSetter(true, true, true), 0.30f},
        {RandomStateSetter(true, true, false), 0.30f}
    };

    config.num_envs = 256;
    config.num_threads = 12;
    config.ticks_per_step = 8;

    config.target_steps_per_update = 50'000;
    config.num_minibatches = 2;

    config.ac_cfg.shared_layers = { 512, 512 };
    config.ac_cfg.actor_layers  = { 256, 256 };
    config.ac_cfg.critic_layers = { 512, 512 };
    config.ac_cfg.use_layer_norm = true;

    config.ppo_cfg.policy_lr = 3e-4f;
    config.ppo_cfg.critic_lr = 3e-4f;
    config.ppo_cfg.entropy_coef = 0.035f;
    config.ppo_cfg.epochs = 1;
    config.ppo_cfg.target_kl = 0.015f;

    PPORunner runner(config);
    runner.run(5'000, g_stop_training, "checkpoints");

    return 0;
}