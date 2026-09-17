// buta-ppo/src/env/reward_manager.hpp
#pragma once

#include "buta_ppo/ffi.h"
#include <vector>
#include <memory>
#include <cmath>

namespace buta_ppo::env {

// Forward declaration of AgentMeta from rocketsim_env.
struct AgentMeta;

// Template function all other functions will extend.
class RewardFunction {
public:
    virtual ~RewardFunction() = default;

    // Called at the start of an episode.
    virtual void reset([[maybe_unused]] const ffi::ArenaState& initial_state) {}

    // Called every tick. Receives the current state and the previous state.
    virtual float get_reward(
        const AgentMeta& agent, 
        const ffi::ArenaState& current_state,
        [[maybe_unused]] const ffi::ArenaState& previous_state
    ) = 0;
};

class TouchBallReward : public RewardFunction {
public:
    float get_reward(const AgentMeta& agent, const ffi::ArenaState& state, const ffi::ArenaState& prev_state) override;
};

class GoalReward : public RewardFunction {
public:
    float get_reward(const AgentMeta& agent, const ffi::ArenaState& state, const ffi::ArenaState& prev_state) override;
};

class VelocityToBallReward : public RewardFunction {
public:
    float get_reward(const AgentMeta& agent, const ffi::ArenaState& state, const ffi::ArenaState& prev_state) override;
};

class RewardManager {
private:
    struct RewardEntry {
        std::unique_ptr<RewardFunction> function;
        float weight;
    };

    std::vector<RewardEntry> rewards_;
    ffi::ArenaState previous_state_;

public:
    RewardManager() = default;

    // Register a reward function with a specific weight.
    void add_reward(std::unique_ptr<RewardFunction> reward_func, float weight = 1.0f);

    // Resets all reward trackers and caches the initial state.
    void reset(const ffi::ArenaState& initial_state);

    // Calculates the weighted sum of all rewards for a given agent.
    float get_reward(const AgentMeta& agent, const ffi::ArenaState& current_state);

    // Updates the previous state. Called after all rewards for a tick have been fetched.
    void update_previous_state(const ffi::ArenaState& arena_state);
};

} // namespace buta_ppo::env