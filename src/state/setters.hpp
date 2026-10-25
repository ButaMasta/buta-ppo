// buta-ppo/src/state/setters.hpp
#pragma once

#include "state_setter.hpp"

#include <memory>

namespace buta_ppo::state {

std::unique_ptr<StateSetter> create_default_kickoff_setter();
std::unique_ptr<StateSetter> create_random_state_setter(bool rand_ball_speed, bool rand_car_speed, bool cars_on_ground);
std::unique_ptr<StateSetter> create_eval_kickoff_setter(uint32_t bank_seed, uint32_t index, float pos_jitter, float yaw_jitter);

} // namespace buta_ppo::state