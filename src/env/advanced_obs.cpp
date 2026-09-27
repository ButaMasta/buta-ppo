// buta-ppo/src/env/advanced_obs.cpp

#include "advanced_obs.hpp"
#include "rocketsim_env.hpp" // For AgentMeta
#include "math_utils.hpp"
#include <algorithm>
#include <cstddef>

namespace buta_ppo::env {

AdvancedObs::AdvancedObs(size_t max_players_per_team, uint32_t seed)
    : max_players_per_team_(max_players_per_team), rng_(seed), tick_last_updated_ball_pred_(9999) {
    ball_pred_arena_ = ffi::create_arena_ball_sim(0);
    team_A_indices_.reserve(max_players_per_team_);
    team_B_indices_.reserve(max_players_per_team_);
}

size_t AdvancedObs::get_obs_size() const {
    return BALL_OBS + (BALL_OBS * BALL_PRED_TICKS_COUNT) + BOOST_PAD_OBS + AGENT_CAR_OBS + (OTHER_CAR_OBS * max_players_per_team_ * 2 - 1);
}

void AdvancedObs::pre_step_rand(const std::vector<AgentMeta>& agents) {
    team_A_indices_.clear();
    team_B_indices_.clear();

    for (const AgentMeta& agent : agents) {
        if (agent.team == ffi::Team::Blue) {
            team_A_indices_.push_back(agent.car_id);
        } else {
            team_B_indices_.push_back(agent.car_id);
        }
    }

    std::shuffle(team_A_indices_.begin(), team_A_indices_.end(), rng_);
    std::shuffle(team_B_indices_.begin(), team_B_indices_.end(), rng_);
}

bool AdvancedObs::exceeds_moe(const float& a, const float& b, const float& moe) const {
    return std::abs(a - b) > moe;
}

bool AdvancedObs::needs_repred(const ffi::BallSimBallState& curr_ball_sim_state) const {
    const ffi::BallSimPhysState& curr_phys = curr_ball_sim_state.phys;
    const ffi::BallSimPhysState& valid_phys = state_to_verify_.phys;

    return exceeds_moe(curr_phys.pos[0], valid_phys.pos[0], POS_MOE) ||
           exceeds_moe(curr_phys.pos[1], valid_phys.pos[1], POS_MOE) ||
           exceeds_moe(curr_phys.pos[2], valid_phys.pos[2], POS_MOE) ||
           exceeds_moe(curr_phys.vel[0], valid_phys.vel[0], VEL_MOE) ||
           exceeds_moe(curr_phys.vel[1], valid_phys.vel[1], VEL_MOE) ||
           exceeds_moe(curr_phys.vel[2], valid_phys.vel[2], VEL_MOE) ||
           exceeds_moe(curr_phys.ang_vel[0], valid_phys.ang_vel[0], ANG_VEL_MOE) ||
           exceeds_moe(curr_phys.ang_vel[1], valid_phys.ang_vel[1], ANG_VEL_MOE) ||
           exceeds_moe(curr_phys.ang_vel[2], valid_phys.ang_vel[2], ANG_VEL_MOE);
}

void AdvancedObs::pred_ballsim(const ffi::BallSimBallState& curr_arena_ball_state, uint64_t ticks_at_update) {
    if (ticks_at_update == tick_last_updated_ball_pred_) { return; }

    if (needs_repred(curr_arena_ball_state)) {
        ffi::set_ball_state_ball_sim(ball_pred_arena_, curr_arena_ball_state);
        ball_pred_head_ = 0;

        for (size_t i = 0; i < BALL_PRED_BUFFER_SIZE; i++) {
            ffi::step_arena_ball_sim(ball_pred_arena_, 8);
            ffi::get_ball_sim_arena_ball_state(ball_pred_arena_, ball_pred_[i]);
        }
    } else {
        ffi::step_arena_ball_sim(ball_pred_arena_, 8);
        ffi::get_ball_sim_arena_ball_state(ball_pred_arena_, ball_pred_[ball_pred_head_]);

        ball_pred_head_ = (ball_pred_head_ + 1) % BALL_PRED_BUFFER_SIZE;
    }

    state_to_verify_ = ball_pred_[ball_pred_head_];
    tick_last_updated_ball_pred_ = ticks_at_update;
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

void AdvancedObs::write_ball_pred(float*& ptr, const float* agent_pos, const float* agent_vel, bool invert_team, bool invert_x) {
    for (size_t i = 0; i < BALL_PRED_TICKS_COUNT; i++) {
        size_t ring_buff_idx = (ball_pred_head_ + BALL_PRED_TICKS[i] - 1) % BALL_PRED_BUFFER_SIZE;
        const ffi::BallSimPhysState& curr_pred_phys = ball_pred_[ring_buff_idx].phys;
        
        write_pos(ptr, curr_pred_phys.pos, invert_team, invert_x);

        float rel_pos[3];
        math::sub_vec3(curr_pred_phys.pos, agent_pos, rel_pos);
        write_pos(ptr, rel_pos, invert_team, invert_x);

        write_vel(ptr, curr_pred_phys.vel, invert_team, invert_x);
        
        float rel_vel[3];
        math::sub_vec3(curr_pred_phys.vel, agent_vel, rel_vel);
        write_vel(ptr, rel_vel, invert_team, invert_x);

        write_ang_vel(ptr, curr_pred_phys.ang_vel, invert_team, invert_x);
    }
}

void AdvancedObs::write_car(float*& ptr, const ffi::ArenaState& arena_state, uint32_t target_car_id, const float* agent_pos, const float* agent_vel, bool invert_team, bool invert_x, bool is_agent) const {
    const auto& car = arena_state.cars[target_car_id];
    size_t obs_size = is_agent ? AGENT_CAR_OBS : OTHER_CAR_OBS;

    // Pad with zeros if the car is demoed and leave only the respawn timer.
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
        float rel_pos[3];
        math::sub_vec3(phys.pos, agent_pos, rel_pos);
        write_pos(ptr, rel_pos, invert_team, invert_x);
    }

    write_dir(ptr, phys.rot_mat[0], invert_team, invert_x);
    write_right_dir(ptr, phys.rot_mat[1], invert_team, invert_x);
    write_dir(ptr, phys.rot_mat[2], invert_team, invert_x);
    write_vel(ptr, phys.vel, invert_team, invert_x);

    if (!is_agent) {
        float rel_vel[3];
        math::sub_vec3(phys.vel, agent_vel, rel_vel);
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

    float rel_pos[3];
    math::sub_vec3(ball_phys.pos, agent_pos, rel_pos);
    write_pos(ptr, rel_pos, invert_team, invert_x);
    
    write_vel(ptr, ball_phys.vel, invert_team, invert_x);
    
    float rel_vel[3];
    math::sub_vec3(ball_phys.vel, agent_vel, rel_vel);
    write_vel(ptr, rel_vel, invert_team, invert_x);
    
    write_ang_vel(ptr, ball_phys.ang_vel, invert_team, invert_x);

    // Ball Pred.
    const ffi::BallState& arena_ball = arena_state.ball;
    ffi::BallSimBallState curr_state = {
        {
            { arena_ball.phys.pos[0], arena_ball.phys.pos[1], arena_ball.phys.pos[2] },
            { arena_ball.phys.vel[0], arena_ball.phys.vel[1], arena_ball.phys.vel[2] },
            { arena_ball.phys.ang_vel[0], arena_ball.phys.ang_vel[1], arena_ball.phys.ang_vel[2] }
        }
    };
    pred_ballsim(curr_state, arena_state.tick_count);
    write_ball_pred(ptr, agent_pos, agent_vel, invert_team, invert_x);

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

    const auto& teammate_list = (agent.team == ffi::Team::Blue) ? team_A_indices_ : team_B_indices_;
    const auto& opponent_list = (agent.team == ffi::Team::Blue) ? team_B_indices_ : team_A_indices_;

    // Teammate Cars.
    size_t teammate_count = 0;
    for (uint32_t other_car_id : teammate_list) {
        if (other_car_id == agent.car_id) continue;
        write_car(ptr, arena_state, other_car_id, agent_pos, agent_vel, invert_team, invert_x, false);
        teammate_count++;
    }
    for (size_t i = teammate_count; i < max_players_per_team_ - 1; i++) {
        write_empty_car(ptr);
    }

    // Opponent Cars.
    size_t opponent_count = 0;
    for (uint32_t other_car_id : opponent_list) {
        write_car(ptr, arena_state, other_car_id, agent_pos, agent_vel, invert_team, invert_x, false);
        opponent_count++;
    }
    for (size_t i = opponent_count; i < max_players_per_team_; i++) {
        write_empty_car(ptr);
    }

    return invert_x;
}

} // namespace buta_ppo::env