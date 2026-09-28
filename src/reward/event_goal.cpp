// buta-ppo/src/reward/touch_ball.cpp
#include "reward_manager.hpp"
#include "rewards.hpp"
#include "env/rocketsim_env.hpp"

#include <memory>

namespace buta_ppo::reward {

/**
 * @brief Rewards the bot for scoring a goal, punishes them for being scored on.
 */
class GoalReward : public RewardFunction {
public:
    float get_reward(const env::AgentMeta& agent, const ffi::ArenaState& state, const ffi::ArenaState& /*prev_state*/) override {
        if (!state.events.is_ball_scored) {
            return 0.0f;
        }
        bool blue_scored = state.ball.phys.pos[1] > 0;
        if ((agent.team == ffi::Team::Blue && blue_scored) || (agent.team == ffi::Team::Orange && !blue_scored)) {
            return 1.0f;
        }

        return -1.0f;
    }
};

std::unique_ptr<RewardFunction> create_goal_reward() {
    return std::make_unique<GoalReward>();
}

} // namespace buta_ppo::reward 
