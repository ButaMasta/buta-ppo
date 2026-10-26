// buta-ppo/src/reward/velocity_ball_to_goal.cpp
#include "reward_manager.hpp"
#include "rewards.hpp"

#include <cmath>
#include <memory>

namespace buta_ppo::reward {

/**
 * @brief Rewards the team based on the ball's velocity towards the opponent's net, punishes velocity away from it.
 */
class VelocityBallToGoalReward : public TeamRewardFunction {
public:
    float get_team_reward(ffi::Team team, const ffi::ArenaState& state, const ffi::ArenaState& /*prev_state*/) override {
        const auto& ball_phys = state.ball.phys;

        // Aim at the back of the net rather than the goal line, so the direction stays stable when the ball is close to the goal.
        constexpr float GOAL_BACK_Y = 6000.0f;
        constexpr float GOAL_CENTER_Z = 642.775f / 2.0f;
        // Blue attacks orange's net (positive Y).
        const float goal_y = team == ffi::Team::Blue ? GOAL_BACK_Y : -GOAL_BACK_Y;

        // Dir to goal.
        float dir_x = 0.0f - ball_phys.pos[0];
        float dir_y = goal_y - ball_phys.pos[1];
        float dir_z = GOAL_CENTER_Z - ball_phys.pos[2];

        // Dist to goal.
        float distance = std::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);

        if (distance < 0.001f) return 0.0f;

        // Norm dir.
        dir_x /= distance;
        dir_y /= distance;
        dir_z /= distance;

        // Magnitude of ball's curr vel to goal.
        float dot_product = (dir_x * ball_phys.vel[0]) +
                            (dir_y * ball_phys.vel[1]) +
                            (dir_z * ball_phys.vel[2]);

        // Normalize by the ball's max speed.
        constexpr float BALL_MAX_SPEED = 6000.0f;
        return dot_product / BALL_MAX_SPEED;
    }

    [[nodiscard]] virtual std::unique_ptr<RewardFunction> clone() const override {
        return std::make_unique<VelocityBallToGoalReward>(*this);
    }
};

std::unique_ptr<RewardFunction> create_velocity_ball_to_goal_reward() {
    return std::make_unique<VelocityBallToGoalReward>();
}

} // namespace buta_ppo::reward
