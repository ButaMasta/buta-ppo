// buta-ppo/src/env/advanced_obs.hpp

/*
*   Implementation is entirely based off of advanced obs used in rlgymppo-rs and GigaLearn.
*/

#pragma once

#include "buta_ppo/ffi.h"
#include <vector>
#include <cstdint>
#include <random>

namespace buta_ppo::env {

// Forward declaration of AgentMeta.
struct AgentMeta;

class AdvancedObs {
public:
    static constexpr size_t CAR_OBS = 22;
    static constexpr size_t BALL_OBS = 9;
    static constexpr size_t BOOST_PAD_OBS = 34;

    static constexpr size_t BOOST_PADS_BIG_COUNT = 6;
    static constexpr size_t BOOST_PADS_SMALL_COUNT = 28;

    static constexpr float POS_COEF_X = 1.0f / 4096.0f;
    static constexpr float POS_COEF_Y = 1.0f / 6000.0f;
    static constexpr float POS_COEF_Z = 1.0f / 2044.0f;
    static constexpr float VEL_COEF = 1.0f / 6000.0f;
    static constexpr float ANG_VEL_COEF = 1.0f / 6.0f;
    static constexpr float BOOST_COEF = 1.0f / 100.0f;
    static constexpr float DEMO_COEF = 1.0f / 3.0f;
    static constexpr float COOLDOWN_BIG_COEF = 1.0f / 10.0f;
    static constexpr float COOLDOWN_SMALL_COEF = 1.0f / 4.0f;

    explicit AdvancedObs(size_t max_players_per_team = 4, uint32_t seed = std::random_device{}());

    // Returns total floats per agent.
    [[nodiscard]] size_t get_obs_size() const;

    // Writes observation in-place.
    void build_obs(
        const ffi::ArenaState& arena_state,
        const std::vector<AgentMeta>& agents,
        uint32_t agent_idx,
        float* out_buffer
    );

private:
    size_t max_players_per_team_;
    std::mt19937 rng_;

    std::vector<uint32_t> teammate_indices_;
    std::vector<uint32_t> opponent_indices_;

    // Normalization & writer helpers.
    void write_vec3_norm(float*& ptr, const float* vec, bool invert, float cx, float cy, float cz) const;
    void write_vec3_norm(float*& ptr, const float* vec, bool invert, float c) const;
    void write_vec3_dir(float*& ptr, const float* vec, bool invert) const;
    void write_car(float*& ptr, const ffi::ArenaState& arena_state, uint32_t target_car_id, bool invert) const;
    void write_empty_car(float*& ptr) const;
};

} // namespace buta_ppo::env