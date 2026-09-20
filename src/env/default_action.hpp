// buta-ppo/src/env/default_action.hpp

/*
*   Implementation is entirely based off of actions used in rlgymppo_rs and GigaLearn.
*/

#pragma once

#include "buta_ppo/ffi.h"
#include <array>
#include <cstdint>

namespace buta_ppo::env {

constexpr size_t ACTION_SPACE_SIZE = 90;

class DefaultAction {
private:
    std::array<ffi::CarControls, ACTION_SPACE_SIZE> actions_table_;
    std::array<bool, ACTION_SPACE_SIZE> ground_mask_;
    std::array<bool, ACTION_SPACE_SIZE> air_mask_;
    std::array<bool, ACTION_SPACE_SIZE> jump_mask_;
    std::array<bool, ACTION_SPACE_SIZE> boost_mask_;

public:
    DefaultAction();

    // Retrieves the physical CarControls for a given discrete index.
    ffi::CarControls get_action(int action_index) const;

    // Writes a validity mask (1.0f for valid, 0.0f for invalid) into out_mask. NOTE: Must have ACTION_SPACE_SIZE space.
    void get_action_mask(const ffi::CarState& car_state, float* out_mask) const;

    constexpr size_t get_action_space_size() const { return ACTION_SPACE_SIZE; }
};

} // namespace buta_ppo::env