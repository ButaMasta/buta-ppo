// buta-ppo/src/reward/rewards.hpp
#pragma once

#include "reward_manager.hpp"

#include <memory>

namespace buta_ppo::reward {

std::unique_ptr<RewardFunction> create_touch_ball_reward();
std::unique_ptr<RewardFunction> create_velocity_to_ball_reward();
std::unique_ptr<RewardFunction> create_goal_reward();

} // namespace buta_ppo::reward 
