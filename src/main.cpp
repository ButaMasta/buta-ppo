// buta-ppo/src/main.cpp
#include "rl/ppo_runner.hpp"
#include <iostream>
#include <csignal>
#include <atomic>
#include <iostream>
#include <locale>

namespace ffi = buta_ppo::ffi;
using namespace buta_ppo::rl;

std::atomic<bool> g_stop_training{false};

void handle_sigint(int) {
    g_stop_training = true;
}

int main() {
    std::cout.imbue(std::locale("en_US.utf8"));
    std::signal(SIGINT, handle_sigint);

    if (!ffi::init("assets/collision_meshes", true)) {
        std::cerr << "[ERROR] Failed to initialize RocketSim." << std::endl;
        return 1;
    }

    RunnerConfig config;

    config.bot_name = "default";
    config.render = true;

    config.num_envs = 256;
    config.agents_per_env = 2;
    config.ticks_per_step = 8;

    config.target_steps_per_update = 50'000;
    config.num_minibatches = 4;

    config.ac_cfg.shared_layers = { 1024, 768 };
    config.ac_cfg.actor_layers  = { 512, 512 };
    config.ac_cfg.critic_layers = { 1024, 1024 };
    config.ac_cfg.use_layer_norm = true;

    config.ppo_cfg.policy_lr = 3e-4f;
    config.ppo_cfg.critic_lr = 3e-4f;
    config.ppo_cfg.entropy_coef = 0.035f;
    config.ppo_cfg.epochs = 3;
    config.ppo_cfg.target_kl = 0.015f;

    PPORunner runner(config);
    runner.run(1000, g_stop_training, "checkpoints");

    return 0;
}