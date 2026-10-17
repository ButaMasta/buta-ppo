// buta-ppo/src/reward/touch_ball.cpp
#include "reward_manager.hpp"
#include "rewards.hpp"
#include "env/rocketsim_env.hpp"

#include <memory>

namespace buta_ppo::reward {

/**
 * @brief Rewards the team that scored a goal. Pair with `opp_scale = 1` to also punish the team that was scored on.
 */
class GoalReward : public TeamRewardFunction {
public:
    float get_team_reward(ffi::Team team, const ffi::ArenaState& state, const ffi::ArenaState& /*prev_state*/) override {
        if (!state.events.is_ball_scored) {
            return 0.0f;
        }
        // The ball is past orange's goal line (positive Y) when blue scores.
        const ffi::Team scoring_team = state.ball.phys.pos[1] > 0 ? ffi::Team::Blue : ffi::Team::Orange;
        return team == scoring_team ? 1.0f : 0.0f;
    }

    [[nodiscard]] virtual std::unique_ptr<RewardFunction> clone() const override {
        return std::make_unique<GoalReward>(*this);
    }
};

std::unique_ptr<RewardFunction> create_goal_reward() {
    return std::make_unique<GoalReward>();
}

} // namespace buta_ppo::reward 
