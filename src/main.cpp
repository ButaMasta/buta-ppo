// buta-ppo/src/main.cpp
#include "buta_ppo/ffi.h"
#include "env/vec_env.hpp"
#include "env/advanced_obs.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>

namespace ffi = buta_ppo::ffi;
namespace env = buta_ppo::env;

int main() {
    if (!ffi::init("assets/collision_meshes", true)) {
        std::cerr << "[ERROR] Failed to initialize RocketSim." << std::endl;
        return 1;
    }

    constexpr size_t NUM_ENVS = 4;
    constexpr size_t NUM_THREADS = 2;
    
    std::cout << "--- Initializing VecEnv (" << NUM_ENVS << " Envs, " << NUM_THREADS << " Threads) ---" << std::endl;
    env::VecEnv vec_env(NUM_ENVS, NUM_THREADS, 8, 3);

    size_t total_agents = vec_env.get_total_agents();
    size_t single_obs = vec_env.get_single_obs_size();
    
    std::cout << "[INIT] Total Agents: " << total_agents << " (Expected: 8)" << std::endl;
    std::cout << "[INIT] Single Obs Size: " << single_obs << std::endl;

    const auto& batched_obs = vec_env.reset();
    std::cout << "[RESET] Batched Obs Buffer Size: " << batched_obs.size() 
              << " (Expected: " << (total_agents * single_obs) << ")" << std::endl;

    std::vector<int> batched_actions(total_agents, 0);
    std::vector<float> initial_y_positions(total_agents, 0.0f);
    
    size_t self_y_offset = env::AdvancedObs::BALL_OBS + env::AdvancedObs::BOOST_PAD_OBS + 1;
    float denorm_y = 6000.0f;

    for (size_t i = 0; i < total_agents; ++i) {
        batched_actions[i] = (i % 2 == 0) ? 16 : 8; // Blue drives, Orange idles
        initial_y_positions[i] = batched_obs[(i * single_obs) + self_y_offset] * denorm_y;
    }

    std::cout << "\n--- Stepping VecEnv (1 Second) ---" << std::endl;

    for (int step = 1; step < 15; ++step) {
        vec_env.step(batched_actions);
    }
    env::BatchedStepResult result = vec_env.step(batched_actions);

    std::cout << std::left << std::setw(10) << "Agent ID" 
              << std::setw(10) << "Team" 
              << std::setw(15) << "Action Given" 
              << std::setw(15) << "Tick Reward" 
              << std::setw(15) << "Delta Y" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (size_t i = 0; i < total_agents; ++i) {
        bool is_blue = (i % 2 == 0);
        int action = batched_actions[i];
        float reward = result.rewards[i];
        
        float final_y = result.observations[(i * single_obs) + self_y_offset] * denorm_y;
        float delta_y = std::abs(final_y - initial_y_positions[i]);

        std::cout << std::left << std::setw(10) << i 
                  << std::setw(10) << (is_blue ? "Blue" : "Orange")
                  << std::setw(15) << action 
                  << std::setw(15) << std::fixed << std::setprecision(5) << reward 
                  << std::setw(15) << delta_y << std::endl;

        if (is_blue) {
            assert(delta_y > 20.0f);
        } else {
            assert(delta_y < 2.0f);
        }
    }

    std::cout << "\n[SUCCESS] Tests complete." << std::endl;

    return 0;
}