// buta-ppo/src/env/default_action.hpp

/*
*   Implementation is based off of actions used in GigaLearn.
*
*/

#pragma once

#include "buta_ppo/ffi.h"

#include <array>
#include <cstddef>

namespace buta_ppo::env {

class DefaultAction {
private:
    static constexpr size_t ACTION_SPACE_SIZE = 90;

    std::array<ffi::CarControls, ACTION_SPACE_SIZE> actions_table_;
    std::array<bool, ACTION_SPACE_SIZE> ground_mask_;
    std::array<bool, ACTION_SPACE_SIZE> air_mask_;
    std::array<bool, ACTION_SPACE_SIZE> jump_mask_;
    std::array<bool, ACTION_SPACE_SIZE> boost_mask_;

public:
    DefaultAction();

    /**
     * @brief Retrieves the physical CarControls for a given discrete index.
     * 
     * @param action_index The index into the discrete action space for what action to perform.
     * @return ffi::CarControls The translated car controls of that action.
     */
    [[nodiscard]] ffi::CarControls get_action(size_t action_index) const;

    /**
     * @brief Writes a validity mask (1.0f for valid, 0.0f for invalid) into out_mask. 
     * 
     * NOTE: Must have ACTION_SPACE_SIZE space.
     * 
     * @param car_state The state of the car to get the action mask for.
     * @param out_mask The pointer to where to write the action mask.
     */
    void get_action_mask(const ffi::CarState& car_state, float* out_mask) const;

    [[nodiscard]] constexpr size_t get_action_space_size() const { return ACTION_SPACE_SIZE; }
};

} // namespace buta_ppo::env