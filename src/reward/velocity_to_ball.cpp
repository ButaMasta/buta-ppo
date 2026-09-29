// buta-ppo/src/reward/touch_ball.cpp
#include "reward_manager.hpp"
#include "rewards.hpp"
#include "env/rocketsim_env.hpp"

#include <cmath>
#include <memory>

namespace buta_ppo::reward {

/**
 * @brief Rewards the bot based on a factor of their speed to the ball, punishes them for negative velocity to the ball.
 */
class VelocityToBallReward : public RewardFunction {
public:
    float get_reward(const env::AgentMeta& agent, const ffi::ArenaState& state, const ffi::ArenaState& /*prev_state*/) override {
        const auto& car_phys = state.cars[agent.car_id].phys;
        const auto& ball_phys = state.ball.phys;

        // Dir to ball.
        float dir_x = ball_phys.pos[0] - car_phys.pos[0];
        float dir_y = ball_phys.pos[1] - car_phys.pos[1];
        float dir_z = ball_phys.pos[2] - car_phys.pos[2];

        // Dist to ball.
        float distance = std::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);

        if (distance < 0.001f) return 0.0f;

        // Norm dir.
        dir_x /= distance;
        dir_y /= distance;
        dir_z /= distance;

        // Magnitude of car's curr vel to ball.
        float dot_product = (dir_x * car_phys.vel[0]) + 
                            (dir_y * car_phys.vel[1]) + 
                            (dir_z * car_phys.vel[2]);

        // Normalize by the car's max speed.
        constexpr float CAR_MAX_SPEED = 2300.0f;
        return dot_product / CAR_MAX_SPEED;
    }

    [[nodiscard]] virtual std::unique_ptr<RewardFunction> clone() const override {
        return std::make_unique<VelocityToBallReward>(*this);
    }
};

std::unique_ptr<RewardFunction> create_velocity_to_ball_reward() {
    return std::make_unique<VelocityToBallReward>();
}

} // namespace buta_ppo::reward 
