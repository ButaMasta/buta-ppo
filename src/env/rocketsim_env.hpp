// buta-ppo/src/env/rocketsim_env.hpp
#pragma once

#include "buta_ppo/ffi.h"
#include "advanced_obs.hpp"
#include <vector>
#include <cstdint>

namespace buta_ppo::env {

struct StepResult {
    // Every agent's observations.
    const std::vector<float>& observations;
    // Every agent's accumulated rewards.
    const std::vector<float>& rewards;
    // If the step is complete, a terminal state was triggered.
    bool is_done;
    // Ticks elapsed this step. Can be lower if a terminal state was triggered.
    int ticks_elapsed;
};

struct AgentMeta {
    uint32_t car_id;
    ffi::Team team;
};

class RocketSimEnv {
private:
    ffi::ArenaPtr arena_;
    ffi::ArenaState arena_state_;

    std::vector<AgentMeta> agents_;

    std::vector<float> obs_buffer_;
    std::vector<float> reward_buffer_;

    int ticks_per_step_;
    size_t single_obs_size_;

    AdvancedObs obs_builder_;

    // Helpers.
    ffi::CarControls decode_action(int action_idx);
    float calculate_reward(uint32_t agent_idx, const ffi::ArenaState& arena_state);

public:
    explicit RocketSimEnv(
        int ticks_per_step = 8, 
        size_t single_obs_size = 45,
        uint32_t seed = std::random_device{}()
    );

    // Add an agent to the env.
    uint32_t add_agent(ffi::Team team);

    // Reset the episode.
    const std::vector<float>& reset();

    // Step environment and update obs and reward buffers for collection.
    StepResult step(const std::vector<int>& actions);

    [[nodiscard]] size_t get_obs_size() const { return single_obs_size_; }
};

}; // namespace buta_ppo::env