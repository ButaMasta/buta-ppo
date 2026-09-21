// buta-ppo/src/env/rocketsim_env.hpp
#pragma once

#include "buta_ppo/ffi.h"
#include "advanced_obs.hpp"
#include "reward_manager.hpp"
#include "default_action.hpp"
#include <vector>
#include <cstdint>

namespace buta_ppo::env {

struct ResetResult {
    const std::vector<float>& observations;
    const std::vector<float>& action_masks;
};

struct StepResult {
    // Every agent's observations.
    const std::vector<float>& observations;
    // Every agent's action_masks.
    const std::vector<float>& action_masks;
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
    std::vector<float> action_mask_buffer_;
    std::vector<float> reward_buffer_;

    int ticks_per_step_;
    size_t single_obs_size_;
    size_t action_space_size_;
    static constexpr int ticks_until_terminal_state_ = 1200;
    int ticks_since_last_touch_ = 0;

    AdvancedObs obs_builder_;
    RewardManager reward_manager_;
    DefaultAction action_parser_;

    bool render_;

    // Helpers.
    ffi::CarControls decode_action(int action_idx);

public:
    explicit RocketSimEnv(
        int ticks_per_step = 8, 
        size_t max_players_per_team = 3,
        uint32_t seed = std::random_device{}(),
        bool render = false
    );

    // Add an agent to the env.
    uint32_t add_agent(ffi::Team team);

    // Reset the episode.
    ResetResult reset();

    // Step environment and update obs and reward buffers for collection.
    StepResult step(const int* actions);

    [[nodiscard]] size_t get_obs_size() const { return single_obs_size_; }
    [[nodiscard]] size_t get_action_space_size() const { return action_space_size_; }
};

}; // namespace buta_ppo::env