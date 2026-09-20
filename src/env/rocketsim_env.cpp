// buta-ppo/src/env/rocketsim_env.hpp
#include "rocketsim_env.hpp"
#include <algorithm>

namespace buta_ppo::env {

// Public methods.
RocketSimEnv::RocketSimEnv(int ticks_per_step, size_t max_players_per_team, uint32_t seed)
    : ticks_per_step_(ticks_per_step), obs_builder_(max_players_per_team, seed) {
    arena_ = ffi::create_arena(0);
    single_obs_size_ = obs_builder_.get_obs_size();

    // Setup Rewards.
    reward_manager_.add_reward(std::make_unique<VelocityToBallReward>(), 0.1f);
    reward_manager_.add_reward(std::make_unique<TouchBallReward>(), 1.0f);
    reward_manager_.add_reward(std::make_unique<GoalReward>(), 10.0f);
}

uint32_t RocketSimEnv::add_agent(ffi::Team team) {
    uint32_t car_id = ffi::add_car(arena_, team);
    agents_.push_back({car_id, team});

    // Resize buffers for the new agent.
    obs_buffer_.resize(agents_.size() * single_obs_size_, 0.0f);
    reward_buffer_.resize(agents_.size(), 0.0f);

    return car_id;
}

const std::vector<float>& RocketSimEnv::reset() {
    ffi::reset_to_random_kickoff(arena_);
    ffi::get_arena_state(arena_, arena_state_);
    reward_manager_.reset(arena_state_);

    for (size_t i = 0; i < agents_.size(); i++) {
        float* agent_obs_ptr = obs_buffer_.data() + (i * single_obs_size_);
        obs_builder_.build_obs(arena_state_, agents_, static_cast<uint32_t>(i), agent_obs_ptr);
    }
    return obs_buffer_;
}

StepResult RocketSimEnv::step(const int* actions) {
    for (size_t i = 0; i < agents_.size(); i++) {
        ffi::CarControls controls = decode_action(actions[i]);
        ffi::set_car_controls(arena_, agents_[i].car_id, controls);
    }

    std::fill(reward_buffer_.begin(), reward_buffer_.end(), 0.0f);
    int ticks_elapsed = ticks_per_step_;
    bool episode_terminated = false;

    for (int i = 0; i < ticks_per_step_; i++) {
        ffi::step_arena(arena_);
        ffi::get_arena_state(arena_, arena_state_);

        // Calculate and accumulate rewards.
        for (size_t a = 0; a < agents_.size(); a++) {
            reward_buffer_[a] += reward_manager_.get_reward(agents_[a], arena_state_);
        }

        if (arena_state_.events.is_ball_scored) {
            ticks_elapsed = i + 1;
            episode_terminated = true;
            break;
        }
    }

    for (size_t i = 0; i < agents_.size(); i++) {
        float* agent_obs_ptr = obs_buffer_.data() + (i * single_obs_size_);
        obs_builder_.build_obs(arena_state_, agents_, static_cast<uint32_t>(i), agent_obs_ptr);
    }

    return { obs_buffer_, reward_buffer_, episode_terminated, ticks_elapsed };
}

// Helpers.
ffi::CarControls RocketSimEnv::decode_action(int action_idx) {
    return action_parser_.get_action(action_idx);
}

}; // namespace buta_ppo::env