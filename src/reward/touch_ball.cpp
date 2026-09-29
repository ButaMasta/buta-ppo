// buta-ppo/src/reward/touch_ball.cpp
#include "reward_manager.hpp"
#include "rewards.hpp"
#include "env/rocketsim_env.hpp"

#include <memory>

namespace buta_ppo::reward {

/**
    * @brief Rewards the bot for touching the ball.
    */
class TouchBallReward : public RewardFunction {
public:
    float get_reward(const env::AgentMeta& agent, const ffi::ArenaState& state, const ffi::ArenaState& /*prev_state*/) override {
        if (state.events.car_hit_ball[agent.car_id]) {
            return 1.0f;
        }
        return 0.0f;
    }

    [[nodiscard]] virtual std::unique_ptr<RewardFunction> clone() const override {
        return std::make_unique<TouchBallReward>(*this);
    }
};

std::unique_ptr<RewardFunction> create_touch_ball_reward() {
    return std::make_unique<TouchBallReward>();
}

} // namespace buta_ppo::reward 
