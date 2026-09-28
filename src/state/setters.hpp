// buta-ppo/src/state/setters.hpp
#pragma once

#include "state_setter.hpp"

#include <memory>

namespace buta_ppo::state {

std::unique_ptr<StateSetter> create_default_kickoff_setter();
std::unique_ptr<StateSetter> create_random_state_setter(bool rand_ball_speed, bool rand_car_speed, bool cars_on_ground);

} // namespace buta_ppo::state