// buta-ppo/src/env/advanced_obs.cpp

#include "advanced_obs.hpp"
#include "rocketsim_env.hpp" // For AgentMeta
#include <algorithm>
#include <cstddef>

namespace buta_ppo::env {

AdvancedObs::AdvancedObs(size_t max_players_per_team, uint32_t seed)
    : max_players_per_team_(max_players_per_team), rng_(seed) {
    teammate_indices_.reserve(max_players_per_team_ > 0 ? max_players_per_team_ - 1 : 0);
    opponent_indices_.reserve(max_players_per_team_);
}

size_t AdvancedObs::get_obs_size() const {
    return BALL_OBS + BOOST_PAD_OBS + AGENT_CAR_OBS + (OTHER_CAR_OBS * max_players_per_team_ * 2 - 1);
}

void AdvancedObs::write_pos(float*& ptr, const float* vec, bool invert_team, bool invert_x) const {
    float x_mult = (invert_team ? -1.0f : 1.0f) * (invert_x ? -1.0f : 1.0f);
    float y_mult = invert_team ? -1.0f : 1.0f;
    *ptr++ = vec[0] * x_mult * POS_COEF_X;
    *ptr++ = vec[1] * y_mult * POS_COEF_Y;
    *ptr++ = vec[2] * POS_COEF_Z;
}

void AdvancedObs::write_vel(float*& ptr, const float* vec, bool invert_team, bool invert_x) const {
    float x_mult = (invert_team ? -1.0f : 1.0f) * (invert_x ? -1.0f : 1.0f);
    float y_mult = invert_team ? -1.0f : 1.0f;
    *ptr++ = vec[0] * x_mult * VEL_COEF;
    *ptr++ = vec[1] * y_mult * VEL_COEF;
    *ptr++ = vec[2] * VEL_COEF;
}

void AdvancedObs::write_ang_vel(float*& ptr, const float* vec, bool invert_team, bool invert_x) const {
    float x_mult = invert_team ? -1.0f : 1.0f;
    float y_mult = (invert_team ? -1.0f : 1.0f) * (invert_x ? -1.0f : 1.0f);
    float z_mult = invert_x ? -1.0f : 1.0f;
    *ptr++ = vec[0] * x_mult * ANG_VEL_COEF;
    *ptr++ = vec[1] * y_mult * ANG_VEL_COEF;
    *ptr++ = vec[2] * z_mult * ANG_VEL_COEF;
}

void AdvancedObs::write_dir(float*& ptr, const float* vec, bool invert_team, bool invert_x) const {
    float x_mult = (invert_team ? -1.0f : 1.0f) * (invert_x ? -1.0f : 1.0f);
    float y_mult = invert_team ? -1.0f : 1.0f;
    *ptr++ = vec[0] * x_mult;
    *ptr++ = vec[1] * y_mult;
    *ptr++ = vec[2];
}

void AdvancedObs::write_right_dir(float*& ptr, const float* vec, bool invert_team, bool invert_x) const {
    float x_mult = invert_team ? -1.0f : 1.0f;
    float y_mult = (invert_team ? -1.0f : 1.0f) * (invert_x ? -1.0f : 1.0f);
    float z_mult = invert_x ? -1.0f : 1.0f;
    *ptr++ = vec[0] * x_mult;
    *ptr++ = vec[1] * y_mult;
    *ptr++ = vec[2] * z_mult;
}

void AdvancedObs::write_car(float*& ptr, const ffi::ArenaState& arena_state, uint32_t target_car_id, const float* agent_pos, const float* agent_vel, bool invert_team, bool invert_x, bool is_agent) const {
    const auto& car = arena_state.cars[target_car_id];
    size_t obs_size = is_agent ? AGENT_CAR_OBS : OTHER_CAR_OBS;

    if (car.is_demoed) {
        for (size_t i = 0; i < obs_size - 1; i++) {
            *ptr++ = 0.0f;
        }
        *ptr++ = car.demo_respawn_timer * DEMO_COEF;
        return;
    }

    const auto& phys = car.phys;
    write_pos(ptr, phys.pos, invert_team, invert_x);

    if (!is_agent) {
        float rel_pos[3] = {
            phys.pos[0] - agent_pos[0],
            phys.pos[1] - agent_pos[1],
            phys.pos[2] - agent_pos[2]
        };
        write_pos(ptr, rel_pos, invert_team, invert_x);
    }

    write_dir(ptr, phys.rot_mat[0], invert_team, invert_x);
    write_right_dir(ptr, phys.rot_mat[1], invert_team, invert_x);
    write_dir(ptr, phys.rot_mat[2], invert_team, invert_x);
    write_vel(ptr, phys.vel, invert_team, invert_x);

    if (!is_agent) {
        float rel_vel[3] = {
            phys.vel[0] - agent_vel[0],
            phys.vel[1] - agent_vel[1],
            phys.vel[2] - agent_vel[2]
        };
        write_vel(ptr, rel_vel, invert_team, invert_x);
    }

    write_ang_vel(ptr, phys.ang_vel, invert_team, invert_x);

    *ptr++ = car.boost * BOOST_COEF;
    *ptr++ = car.is_on_ground ? 1.0f : 0.0f;
    *ptr++ = car.has_flip_or_jump ? 1.0f : 0.0f;
    *ptr++ = car.is_auto_flipping ? 1.0f : 0.0f;
    *ptr++ = car.air_time_since_jump;
    *ptr++ = car.handbreak_val;
    *ptr++ = car.demo_respawn_timer * DEMO_COEF;
}

void AdvancedObs::write_empty_car(float*& ptr) const {
    for (size_t i = 0; i < OTHER_CAR_OBS; i++) {
        *ptr++ = 0.0f;
    }
}

bool AdvancedObs::build_obs(
    const ffi::ArenaState& arena_state,
    const std::vector<AgentMeta>& agents,
    uint32_t agent_idx,
    float* out_buffer
) {
    const AgentMeta& agent = agents[agent_idx];
    float* ptr = out_buffer;

    const auto& agent_phys = arena_state.cars[agent.car_id].phys;
    const float* agent_pos = agent_phys.pos;
    const float* agent_vel = agent_phys.vel;

    bool invert_team = (agent.team == ffi::Team::Orange);

    float perceived_x = agent_pos[0] * (invert_team ? -1.0f : 1.0f);
    bool invert_x = perceived_x < 0.0f;

    // Ball State.
    const auto& ball_phys = arena_state.ball.phys;
    write_pos(ptr, ball_phys.pos, invert_team, invert_x);
    write_vel(ptr, ball_phys.vel, invert_team, invert_x);
    write_ang_vel(ptr, ball_phys.ang_vel, invert_team, invert_x);

    // Boost Pads.
    const auto& pads = arena_state.boost_pads;

    for (size_t i = 0; i < BOOST_PADS_BIG_COUNT; i++) {
        size_t read_idx = i;
        if (invert_team) read_idx = BOOST_PADS_BIG_COUNT - 1 - read_idx;
        if (invert_x) read_idx = BIG_PAD_INVERT_X[read_idx];
        *ptr++ = pads.big[read_idx] * COOLDOWN_BIG_COEF;
    }

    for (size_t i = 0; i < BOOST_PADS_SMALL_COUNT; i++) {
        size_t read_idx = i;
        if (invert_team) read_idx = BOOST_PADS_SMALL_COUNT - 1 - read_idx;
        if (invert_x) read_idx = SMALL_PAD_INVERT_X[read_idx];
        *ptr++ = pads.small[read_idx] * COOLDOWN_SMALL_COEF;
    }

    // Agent Car.
    write_car(ptr, arena_state, agent.car_id, agent_pos, agent_vel, invert_team, invert_x, true);

    // Teammate Cars.
    teammate_indices_.clear();
    for (const auto& other : agents) {
        if (other.car_id != agent.car_id && other.team == agent.team) {
            teammate_indices_.push_back(other.car_id);
        }
    }
    std::shuffle(teammate_indices_.begin(), teammate_indices_.end(), rng_);
    for (uint32_t teammate_id : teammate_indices_) {
        write_car(ptr, arena_state, teammate_id, agent_pos, agent_vel, invert_team, invert_x, false);
    }
    for (size_t i = teammate_indices_.size(); i < max_players_per_team_ - 1; i++) {
        write_empty_car(ptr);
    }

    // Opponent Cars.
    opponent_indices_.clear();
    for (const auto& other : agents) {
        if (other.team != agent.team) {
            opponent_indices_.push_back(other.car_id);
        }
    }
    std::shuffle(opponent_indices_.begin(), opponent_indices_.end(), rng_);
    for (uint32_t opponent_id : opponent_indices_) {
        write_car(ptr, arena_state, opponent_id, agent_pos, agent_vel, invert_team, invert_x, false);
        // write_car(ptr, arena_state, opponent_id, invert);
    }
    for (size_t i = opponent_indices_.size(); i < max_players_per_team_; i++) {
        write_empty_car(ptr);
    }

    return invert_x;
}

} // namespace buta_ppo::env