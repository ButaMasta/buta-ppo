// buta-ppo/src/env/rocketsim_env.hpp
#pragma once

#include "rocketsim_env.hpp"
#include <iostream>

namespace buta_ppo::env {

// Public methods.

RocketSimEnv::RocketSimEnv(int ticks_per_step, size_t single_obs_size)
    : ticks_per_step_(ticks_per_step), single_obs_size_(single_obs_size) {
    arena_ = ffi::create_arena(0);
    teammate_indices_.reserve(MAX_PLAYER_PER_TEAM - 1);
    opponent_indices_.reserve(MAX_PLAYER_PER_TEAM);
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

    for (size_t i = 0; i < agents_.size(); i++) {
        float* agent_obs_ptr = obs_buffer_.data() + (i * single_obs_size_);
        build_obs(i, agent_obs_ptr);
    }
    return obs_buffer_;
}

StepResult RocketSimEnv::step(const std::vector<int>& actions) {
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
            reward_buffer_[a] += calculate_reward(a, arena_state_);
        }

        if (arena_state_.events.is_ball_scored) {
            ticks_elapsed = i;
            episode_terminated = true;
            break;
        }
    }

    for (size_t i = 0; i < agents_.size(); i++) {
        float* agent_obs_ptr = obs_buffer_.data() + (i * single_obs_size_);
        build_obs(i, agent_obs_ptr);
    }

    return { obs_buffer_, reward_buffer_, episode_terminated, ticks_elapsed };
}

// Helpers.

ffi::CarControls RocketSimEnv::decode_action(int action_idx) {
    ffi::CarControls controls{};
    // TODO: Map controls based on popular discrete action setups.
    return controls;
}

void RocketSimEnv::write_vec3_norm(float*& ptr, const float* vec, bool invert, float cx, float cy, float cz) {
    float multiplier = invert ? -1.0f : 1.0f;
    *ptr++ = vec[0] * multiplier * cx; // X
    *ptr++ = vec[1] * multiplier * cy; // Y
    *ptr++ = vec[2] * cz;              // Z
}

void RocketSimEnv::write_vec3_norm(float*& ptr, const float* vec, bool invert, float c) {
    float multiplier = invert ? -1.0f : 1.0f;
    *ptr++ = vec[0] * multiplier * c; // X
    *ptr++ = vec[1] * multiplier * c; // Y
    *ptr++ = vec[2] * c;              // Z
}

void RocketSimEnv::write_vec3_dir(float*& ptr, const float* vec, bool invert) {
    float multiplier = invert ? -1.0f : 1.0f;
    *ptr++ = vec[0] * multiplier; // X
    *ptr++ = vec[1] * multiplier; // Y
    *ptr++ = vec[2];              // Z
}

void RocketSimEnv::write_car(float*& ptr, uint32_t target_idx, bool invert) {
    const auto& car = arena_state_.cars[target_idx];

    if (car.is_demoed) {
        for (size_t i = 0; i < CAR_OBS - 1; i++) {
            *ptr++ = 0.0f;
        }
        *ptr++ = car.demo_respawn_timer * DEMO_COEF;
        return;
    }

    const auto& phys = car.phys;
    write_vec3_norm(ptr, phys.pos, invert, POS_COEF_X, POS_COEF_Y, POS_COEF_Z);
    write_vec3_dir(ptr, phys.rot_mat[0], invert);
    write_vec3_dir(ptr, phys.rot_mat[2], invert);
    write_vec3_norm(ptr, phys.vel, invert, VEL_COEF);
    write_vec3_norm(ptr, phys.ang_vel, invert, ANG_VEL_COEF);

    *ptr++ = car.boost * BOOST_COEF;
    *ptr++ = car.is_on_ground ? 1.0f : 0.0f;
    *ptr++ = car.has_flip_or_jump ? 1.0f : 0.0f;
    *ptr++ = car.is_auto_flipping ? 1.0f : 0.0f;
    *ptr++ = car.air_time_since_jump;
    *ptr++ = car.handbreak_val;
    *ptr++ = car.demo_respawn_timer * DEMO_COEF;
}

void RocketSimEnv::write_empty_car(float*& ptr) {
    for (size_t i = 0; i < CAR_OBS; i++) {
        *ptr++ = 0.0f;
    }
}

std::vector<float> RocketSimEnv::build_obs(uint32_t agent_idx, float* out_buffer) {
    const AgentMeta& agent = agents_[agent_idx];
    float* ptr = out_buffer;

    bool invert = (agent.team == ffi::Team::Orange);

    // Subject to change, this is a basic obs right now.
    // Add ball data.
    const auto& ball_phys = arena_state_.ball.phys;
    write_vec3_norm(ptr, ball_phys.pos, invert, POS_COEF_X, POS_COEF_Y, POS_COEF_Z);
    write_vec3_norm(ptr, ball_phys.vel, invert, VEL_COEF);
    write_vec3_norm(ptr, ball_phys.ang_vel, invert, ANG_VEL_COEF);

    // Add boost pad data.
    if (invert) {
        for (size_t i = BOOST_PAD_OBS - 1; i >= 0; i--) {
            *ptr++ = arena_state_.pads[i] * PAD_TIMER_COEF;
        }
    } else {
        for (size_t i = 0; i < BOOST_PAD_OBS; i++) {
            *ptr++ = arena_state_.pads[i] * PAD_TIMER_COEF;
        }
    }

    // Add agent's car.
    write_car(ptr, agent.car_id, invert);

    // Teammate cars.
    teammate_indices_.clear();
    for (const auto& other : agents_) {
        if (other.car_id != agent.car_id && other.team == agent.team) {
            teammate_indices_.push_back(other.car_id);
        }
    }
    std::shuffle(teammate_indices_.begin(), teammate_indices_.end(), rng_);
    for (uint32_t teammate_idx : teammate_indices_) {
        write_car(ptr, teammate_idx, invert);
    }
    for (size_t i = teammate_indices_.size(); i < MAX_PLAYER_PER_TEAM - 1; i++) {
        write_empty_car(ptr);
    }

    // Opponent cars.
    opponent_indices_.clear();
    for (const auto& other : agents_) {
        if (other.team != agent.team) {
            opponent_indices_.push_back(other.car_id);
        }
    }
    std::shuffle(opponent_indices_.begin(), opponent_indices_.end(), rng_);
    for (uint32_t opponent_idx : opponent_indices_) {
        write_car(ptr, opponent_idx, invert);
    }
    for (size_t i = opponent_indices_.size(); i < MAX_PLAYER_PER_TEAM; i++) {
        write_empty_car(ptr);
    }
}

float RocketSimEnv::calculate_reward(uint32_t agent_idx, const ffi::ArenaState& arena_state) {
    float reward = 0.0f;

    // TODO: Implement actual reward calculations.
    if (arena_state_.events.car_hit_ball[agent_idx]) {
        reward += 1.0f;
    }
    return reward;
}

}; // namespace buta_ppo::env