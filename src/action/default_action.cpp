// buta-ppo/src/env/default_action.cpp

/*
*   Implementation is entirely based off of actions used in rlgymppo_rs and GigaLearn.
*/

#include "default_action.hpp"

namespace buta_ppo::action {

DefaultAction::DefaultAction() {
    ground_mask_.fill(false);
    air_mask_.fill(false);
    jump_mask_.fill(false);
    boost_mask_.fill(false);

    size_t idx = 0;
    const float throttles[] = {-1.0f, 0.0f, 1.0f};
    const float steers[]    = {-1.0f, 0.0f, 1.0f};
    const bool  booleans[]  = {false, true};

    // Ground Actions.
    for (float throttle : throttles) {
        for (float steer : steers) {
            for (bool boost : booleans) {
                for (bool handbrake : booleans) {
                    // Prevent useless throttle when boosting.
                    if (boost && throttle != 1.0f) continue;

                    ffi::CarControls action{};
                    action.throttle = throttle;
                    action.steer = steer;
                    action.pitch = 0.0f;
                    action.yaw = steer;
                    action.roll = 0.0f;
                    action.jump = false;
                    action.boost = boost;
                    action.handbrake = handbrake;

                    actions_table_[idx++] = action;
                }
            }
        }
    }

    const size_t num_ground_actions = idx;

    // Aerial Actions.
    const float pitches[] = {-1.0f, 0.0f, 1.0f};
    const float yaws[]    = {-1.0f, 0.0f, 1.0f};
    const float rolls[]   = {-1.0f, 0.0f, 1.0f};

    for (float pitch : pitches) {
        for (float yaw : yaws) {
            for (float roll : rolls) {
                for (bool jump : booleans) {
                    for (bool boost : booleans) {
                        // Only need roll for sideflip.
                        if (jump && yaw != 0.0f) continue;
                        
                        // Duplicate with ground.
                        if (pitch == 0.0f && roll == 0.0f && !jump) continue;

                        // Enable handbrake for potential wavedashes.
                        const bool handbrake = jump && (pitch != 0.0f || yaw != 0.0f || roll != 0.0f);

                        ffi::CarControls action{};
                        action.throttle = boost ? 1.0f : 0.0f;
                        action.steer = yaw;
                        action.pitch = pitch;
                        action.yaw = yaw;
                        action.roll = roll;
                        action.jump = jump;
                        action.boost = boost;
                        action.handbrake = handbrake;

                        actions_table_[idx++] = action;
                    }
                }
            }
        }
    }

    // Precompute Masks.
    for (size_t i = 0; i < ACTION_SPACE_SIZE; ++i) {
        const auto& action = actions_table_[i];

        if (action.jump) jump_mask_[i] = true;
        if (action.boost) boost_mask_[i] = true;

        if (i < num_ground_actions) {
            ground_mask_[i] = true;
        }

        if (i >= num_ground_actions && !action.jump) {
            air_mask_[i] = true;
        }

        // Ground actions that are also valid in the air.
        if (i < num_ground_actions && 
            action.throttle == (action.boost ? 1.0f : 0.0f) && 
            (action.yaw != 0.0f) == action.handbrake) {
            air_mask_[i] = true;
        }
    }
}

ffi::CarControls DefaultAction::get_action(size_t action_index) const {
    return actions_table_[action_index];
}

void DefaultAction::get_action_mask(const ffi::CarState& car_state, float* out_mask) const {
    bool has_boost = car_state.boost > 0.0f;
    
    // Check if the car's local Z (Up) vector is pointing sharply downward (-Z). Substitute for world contact normal.
    bool is_turtled = (!car_state.is_on_ground) && 
                      (car_state.phys.rot_mat[2][2] < -0.2f) && 
                      (car_state.phys.pos[2] < 50.0f);
    
    // Check if a jump/flip is possible.
    bool can_jump = car_state.has_flip_or_jump || is_turtled;

    for (size_t i = 0; i < ACTION_SPACE_SIZE; ++i) {
        bool valid = car_state.is_on_ground ? ground_mask_[i] : air_mask_[i];

        if (!has_boost && boost_mask_[i]) valid = false;
        if (can_jump && jump_mask_[i]) valid = true;

        out_mask[i] = valid ? 1.0f : 0.0f;
    }
}

} // namespace buta_ppo::env