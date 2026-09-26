// buta-ppo/src/env/advanced_obs.hpp
#pragma once

#include "buta_ppo/ffi.h"

#include <vector>
#include <cstdint>
#include <cstddef>
#include <random>

// Forward declaration.
namespace buta_ppo::env { struct AgentMeta; }

namespace buta_ppo::env {

class AdvancedObs {
public:
    static constexpr size_t AGENT_CAR_OBS = 25; // base features.
    static constexpr size_t OTHER_CAR_OBS = 31; // + rel pos/vel.
    static constexpr size_t BALL_OBS = 15; // + rel pos/vel.
    static constexpr size_t BOOST_PAD_OBS = 34;

    static constexpr size_t BOOST_PADS_BIG_COUNT = 6;
    static constexpr size_t BOOST_PADS_SMALL_COUNT = 28;

    // This is unfortunately the easiest way to do X-axis boost pad mirroring efficiently.
    static constexpr size_t BIG_PAD_INVERT_X[BOOST_PADS_BIG_COUNT] = {1, 0, 3, 2, 5, 4};
    static constexpr size_t SMALL_PAD_INVERT_X[BOOST_PADS_SMALL_COUNT] = {
        0, 2, 1, 4, 3, 5, 7, 6, 9, 8, 11, 10, 12, 14, 13, 15, 
        17, 16, 19, 18, 21, 20, 22, 24, 23, 26, 25, 27
    };

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

    /**
     * @brief Get the size of the obs.
     * 
     * @return size_t - The size of the obs.
     */
    [[nodiscard]] size_t get_obs_size() const;

    /**
     * @brief Writes observation in-place. Returns X-Mirror status of this state.
     * 
     * @param arena_state - The state of the arena containing all values from the FFI needed for obs construction.
     * @param agents - A list of all the agents in this environment and their corresponding information.
     * @param agent_idx - The agent to build the obs for.
     * @param out_buffer - The float buffer to store the output of the obs. In this case it is a pointer to within a larger buffer.
     * @return true - If the agent IS viewing a X-mirrored state. 
     * @return false - If the agent IS NOT viewing a X-mirrored state.
     */
    bool build_obs(
        const ffi::ArenaState& arena_state,
        const std::vector<AgentMeta>& agents,
        uint32_t agent_idx,
        float* out_buffer
    );

private:
    size_t max_players_per_team_;
    std::mt19937 rng_;

    // These are the lightweight buffers to shuffle when inserting cars into the obs.
    std::vector<uint32_t> teammate_indices_;
    std::vector<uint32_t> opponent_indices_;

    // Normalization & writer helpers.
    // All of these methods will, as their name says, write their values into the obs pointer and increments it.
    void write_pos(float*& ptr, const float* vec, bool invert_team, bool invert_x) const;
    void write_vel(float*& ptr, const float* vec, bool invert_team, bool invert_x) const;
    void write_ang_vel(float*& ptr, const float* vec, bool invert_team, bool invert_x) const;
    void write_dir(float*& ptr, const float* vec, bool invert_team, bool invert_x) const;
    void write_right_dir(float*& ptr, const float* vec, bool invert_team, bool invert_x) const;

    void write_car(float*& ptr, const ffi::ArenaState& arena_state, uint32_t target_car_id, const float* agent_pos, const float* agent_vel, bool invert_team, bool invert_x, bool is_agent) const;
    void write_empty_car(float*& ptr) const;
};

} // namespace buta_ppo::env