// buta-ppo/src/env/reward_manager.cpp
#include "reward_manager.hpp"
#include "rocketsim_env.hpp"

namespace buta_ppo::env {

float TouchBallReward::get_reward(const AgentMeta& agent, const ffi::ArenaState& state, [[maybe_unused]] const ffi::ArenaState& prev_state) {
    if (state.events.car_hit_ball[agent.car_id]) {
        return 1.0f;
    }
    return 0.0f;
}

float GoalReward::get_reward(const AgentMeta& agent, const ffi::ArenaState& state, [[maybe_unused]] const ffi::ArenaState& prev_state) {
    if (!state.events.is_ball_scored) {
        return 0.0f;
    }
    bool blue_scored = state.ball.phys.pos[1] > 0;
    if ((agent.team == ffi::Team::Blue && blue_scored) || (agent.team == ffi::Team::Orange && !blue_scored)) {
        return 1.0f;
    } else {
        return -1.0f;
    }
}

float VelocityToBallReward::get_reward(const AgentMeta& agent, const ffi::ArenaState& state, [[maybe_unused]] const ffi::ArenaState& prev_state) {
    const auto& car_phys = state.cars[agent.car_id].phys;
    const auto& ball_phys = state.ball.phys;

    // Dir to ball.
    float dir_x = ball_phys.pos[0] - car_phys.pos[0];
    float dir_y = ball_phys.pos[1] - car_phys.pos[1];
    float dir_z = ball_phys.pos[2] - car_phys.pos[2];

    // Dist to ball.
    float distance = std::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);

    if (distance < 0.001f) return 0.0f;

    // Norm dir.
    dir_x /= distance;
    dir_y /= distance;
    dir_z /= distance;

    // Magnitude of car's curr vel to ball.
    float dot_product = (dir_x * car_phys.vel[0]) + 
                        (dir_y * car_phys.vel[1]) + 
                        (dir_z * car_phys.vel[2]);

    // Normalize by the car's max speed.
    constexpr float CAR_MAX_SPEED = 2300.0f;
    return dot_product / CAR_MAX_SPEED;
}

void RewardManager::add_reward(std::unique_ptr<RewardFunction> reward_func, float weight) {
    rewards_.push_back({std::move(reward_func), weight});
}

void RewardManager::reset(const ffi::ArenaState& initial_state) {
    previous_state_ = initial_state;
    for (auto& entry : rewards_) {
        entry.function->reset(initial_state);
    }
}

float RewardManager::get_reward(const AgentMeta& agent, const ffi::ArenaState& current_state) {
    float total_reward = 0.0f;

    for (const auto& entry : rewards_) {
        total_reward += entry.function->get_reward(agent, current_state, previous_state_) * entry.weight;
    }
    
    return total_reward;
}

void RewardManager::update_previous_state(const ffi::ArenaState& arena_state) {
    previous_state_ = arena_state;
}


} // namespace buta_ppo::env