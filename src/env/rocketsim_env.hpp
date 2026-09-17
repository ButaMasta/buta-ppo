// buta-ppo/src/env/rocketsim_env.hpp
#pragma once

#include "buta_ppo/ffi.h"
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <numeric>

namespace buta_ppo::env {

constexpr size_t CAR_OBS = 22;
constexpr size_t BALL_OBS = 9;
constexpr size_t BOOST_PAD_OBS = 34;

constexpr float POS_COEF_X = 1.0f / 4096.0f;
constexpr float POS_COEF_Y = 1.0f / 6000.0f;
constexpr float POS_COEF_Z = 1.0f / 2044.0f;
constexpr float VEL_COEF = 1.0f / 6000.0f;
constexpr float ANG_VEL_COEF = 1.0f / 6.0f;
constexpr float BOOST_COEF = 1.0f / 100.0f;
constexpr float DEMO_COEF = 1.0f / 3.0f;
constexpr float PAD_TIMER_COEF = 1.0f / 10.0f;

constexpr size_t MAX_PLAYER_PER_TEAM = 4;

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

    std::mt19937 rng_;

    // Obs simplification.
    std::vector<uint32_t> teammate_indices_;
    std::vector<uint32_t> opponent_indices_;

    // Helpers.

    ffi::CarControls decode_action(int action_idx);
    // Write a vec3 while normalizing it and advancing the pointer.
    void write_vec3_norm(float*& ptr, const float* vec, bool invert, float cx, float cy, float cz);
    void write_vec3_norm(float*& ptr, const float* vec, bool invert, float c);
    // Write a vec3 and advance the pointer.
    void write_vec3_dir(float*& ptr, const float* vec, bool invert);
    // Write a car state and advance the pointer.
    void write_car(float*& ptr, uint32_t target_idx, bool invert);
    // Write a blank car state and advance the pointer.
    void write_empty_car(float*& ptr);

    std::vector<float> build_obs(uint32_t agent_idx, float* out_buffer);
    float calculate_reward(uint32_t agent_idx, const ffi::ArenaState& arena_state);

public:
    RocketSimEnv(int ticks_per_step = 8, size_t single_obs_size = 45);

    // Add an agent to the env.
    uint32_t add_agent(ffi::Team team);

    // Reset the episode.
    const std::vector<float>& reset();

    // Step environment and update obs and reward buffers for collection.
    StepResult step(const std::vector<int>& actions);
};

}; // namespace buta_ppo::env