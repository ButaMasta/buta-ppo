// buta-ppo/src/env/reward_manager.cpp
#include "reward_manager.hpp"
#include "env/rocketsim_env.hpp"

namespace buta_ppo::reward {

void RewardManager::set_rewards(const std::vector<RewardEntry>& rewards) {
        rewards_ = rewards;
        for (const RewardEntry& entry : rewards) {
            telemetry_breakdown_[entry.name] = 0.0;
        }
    }

void RewardManager::reset(const ffi::ArenaState& initial_state) {
    previous_state_ = initial_state;
    for (auto& entry : rewards_) {
        entry.function->reset(initial_state);
    }
}

float RewardManager::get_reward(const env::AgentMeta& agent, const ffi::ArenaState& current_state) {
    float total_reward = 0.0f;

    for (auto& entry : rewards_) {
        float weighted_val = entry.function->get_reward(agent, current_state, previous_state_) * entry.weight;
        entry.accumulated_value += weighted_val;
        total_reward += weighted_val;
    }
    
    return total_reward;
}

void RewardManager::update_telemetry() {
    for (auto& entry : rewards_) {
        telemetry_breakdown_[entry.name] = entry.accumulated_value;
        entry.accumulated_value = 0.0;
    }
}

void RewardManager::update_previous_state(const ffi::ArenaState& arena_state) {
    previous_state_ = arena_state;
}

} // namespace buta_ppo::reward