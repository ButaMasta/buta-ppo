// buta-ppo/src/env/rocketsim_env.hpp
#pragma once

#include "buta_ppo/ffi.h"
#include "obs/advanced_obs.hpp"
#include "reward/reward_manager.hpp"
#include "action/default_action.hpp"
#include "state/state_setter.hpp"

#include <unordered_map>
#include <vector>
#include <cstdint>
#include <random>
#include <string>
#include <cstddef>

namespace buta_ppo::env {

/**
 * @brief Struct housing the result of resetting the env.
 */
struct ResetResult {
    const std::vector<float>& observations;
    const std::vector<float>& action_masks;
};

/**
 * @brief Struct housing the result of a step.
 * 
 */
struct StepResult {
    // Every agent's observations.
    const std::vector<float>& observations;
    // Every agent's action_masks.
    const std::vector<float>& action_masks;
    // Every agent's accumulated rewards.
    const std::vector<float>& rewards;
    // If a terminal state was triggered.
    bool is_terminated;
    // If a truncation state was triggered.
    bool is_truncated;
    // Ticks elapsed this step. Can be lower if a terminal/truncation state was triggered.
    int ticks_elapsed;
};

/**
 * @brief Helper struct to consolodate agent info that is needed in numerous locations.
 */
struct AgentMeta {
    uint32_t car_id;
    ffi::Team team;
};

/**
 * @brief Core wrapper managing simulation execution, buffer allocation, and state capture.
 */
class RocketSimEnv {
private:
    ffi::ArenaPtr arena_;
    ffi::ArenaState arena_state_;

    std::vector<AgentMeta> agents_;

    std::vector<float> obs_buffer_;
    std::vector<float> action_mask_buffer_;
    std::vector<float> reward_buffer_;
    std::vector<bool> agent_x_inverted_;

    std::vector<state::StateSetterDistribution> setters_;
    std::discrete_distribution<size_t> setter_selector_;
    std::mt19937 rng_;

    int ticks_per_step_;
    size_t single_obs_size_;
    size_t action_space_size_;
    static constexpr int no_touch_ticks_limit_ = 1200;  // 10 seconds.
    static constexpr int match_ticks_limit_ = 18'000;   // 2.5 minutes.
    int ticks_since_last_touch_ = 0;
    int match_ticks_ = 0;

    obs::AdvancedObs obs_builder_;
    reward::RewardManager reward_manager_;
    action::DefaultAction action_parser_;

    bool render_;

    /**
     * @brief Wrapper method to grab the car controls for an action given by the action parser in this env.
     * 
     * @param action_idx The action parser's discrete action lookup table idx.
     * @return ffi::CarControls The decoded car controls for this action.
     */
    ffi::CarControls decode_action(size_t action_idx);

public:
    explicit RocketSimEnv(
        const std::vector<ffi::Team>& match_layout,
        const std::vector<state::StateSetterDistribution>& setters,
        const std::vector<reward::RewardEntry>& rewards,
        int ticks_per_step = 8, 
        size_t max_players_per_team = 3,
        uint32_t seed = std::random_device{}(),
        bool render = false
    );

    /**
     * @brief Add an agent to the env.
     * 
     * @param team The team to assign them.
     * @return The agent's ID.
     */
    uint32_t add_agent(ffi::Team team);

    /**
     * @brief Reset the episode.
     * 
     * @return The result of the reset in the new arena state.
     */
    [[nodiscard]] ResetResult reset();

    /**
     * @brief Step environment and update obs and reward buffers for collection.
     * 
     * @param actions Pointer to the actions to be taken in this step for the agents.
     * @return The result of the step.
     */
    [[nodiscard]] StepResult step(const int* actions);

    /**
     * @brief Update the internal reward manager's telemetry.
     */
    void update_reward_telemetry() {
        reward_manager_.update_telemetry();
    }

    [[nodiscard]] const std::unordered_map<std::string, double>& get_reward_telemetry() const { return reward_manager_.get_telemetry(); }
    [[nodiscard]] size_t get_obs_size() const { return single_obs_size_; }
    [[nodiscard]] size_t get_action_space_size() const { return action_space_size_; }
};

}; // namespace buta_ppo::env