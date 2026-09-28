// buta-ppo/src/state/random_state.cpp
#include "setters.hpp"
#include "state_setter.hpp"
#include "util/math_utils.hpp"
#include "env/rocketsim_env.hpp"

#include <memory>

namespace math = buta_ppo::util::math;

namespace buta_ppo::state {

/**
 * @brief Creates a random state with configurable values for enabling randomness of: ball speed, car speed, and cars on ground.
 */
class RandomStateSetter : public StateSetter {
private:
    bool rand_ball_speed_;
    bool rand_car_speed_;
    bool cars_on_ground_;

public:
    RandomStateSetter(bool rand_ball_speed, bool rand_car_speed, bool cars_on_ground)
        : rand_ball_speed_(rand_ball_speed), rand_car_speed_(rand_car_speed), cars_on_ground_(cars_on_ground) {}
    
    void apply(
        const ffi::ArenaPtr& arena, 
        ffi::ArenaState& current_state, 
        const std::vector<env::AgentMeta>& agents, 
        std::mt19937& rng
    ) override {
        // Reset to a normal kickoff first, then randomize to ensure a properly reset arena.
        ffi::reset_to_random_kickoff(arena, rng(), true);
        ffi::get_arena_state(arena, current_state);

        // Values used here are known arena and sim constants.
        std::uniform_real_distribution<float> rand_x(-3500.0f, 3500.0f);
        std::uniform_real_distribution<float> rand_y(-4000.0f, 4000.0f);
        // Using 120.0f here as a general guess that safely allows any car rotation.
        // only downside is that is misses out on valid states from 0.0f to 120.0f
        // but realistically this is a non-issue for state space.
        std::uniform_real_distribution<float> rand_z(120.0f, 1820.0f);
        std::uniform_real_distribution<float> rand_pitch(-1.5707f, 1.5707f);
        std::uniform_real_distribution<float> rand_yaw(-3.1415f, 3.1415f);
        std::uniform_real_distribution<float> rand_roll(-3.1415f, 3.1415f);

        // Randomize ball.
        current_state.ball.phys.pos[0] = rand_x(rng);
        current_state.ball.phys.pos[1] = rand_y(rng);
        current_state.ball.phys.pos[2] = rand_z(rng);

        if (rand_ball_speed_) {
            float dir[3];
            math::random_norm_vec(rng, dir);
            std::uniform_real_distribution<float> speed(0.0f, 4000.0f);
            float s = speed(rng);
            current_state.ball.phys.vel[0] = dir[0] * s;
            current_state.ball.phys.vel[1] = dir[1] * s;
            current_state.ball.phys.vel[2] = dir[2] * s;
        } else {
            current_state.ball.phys.vel[0] = 0.0f;
            current_state.ball.phys.vel[1] = 0.0f;
            current_state.ball.phys.vel[2] = 0.0f;
        }

        ffi::set_ball_state(arena, current_state.ball);

        // Randomize cars.
        for (const auto& agent : agents) {
            auto& car = current_state.cars[agent.car_id];

            car.phys.pos[0] = rand_x(rng); 
            car.phys.pos[1] = rand_y(rng);
            car.phys.pos[2] = cars_on_ground_ ? 17.0f : rand_z(rng);

            if (cars_on_ground_) {
                math::euler_to_mat3(0.0f, rand_yaw(rng), 0.0f, car.phys.rot_mat);
                if (rand_car_speed_) {
                    std::uniform_real_distribution<float> speed(-2300.0f, 2300.0f);
                    float s = speed(rng);

                    car.phys.vel[0] = car.phys.rot_mat[0][0] * s;
                    car.phys.vel[1] = car.phys.rot_mat[0][1] * s;
                }
                car.phys.vel[2] = 0;
                car.phys.ang_vel[0] = 0.0f;
                car.phys.ang_vel[1] = 0.0f;
                car.phys.ang_vel[2] = 0.0f;
            } else {
                math::euler_to_mat3(rand_pitch(rng), rand_yaw(rng), rand_roll(rng), car.phys.rot_mat);
                if (rand_car_speed_) {
                    float dir[3];
                    math::random_norm_vec(rng, dir);
                    std::uniform_real_distribution<float> speed(0.0f, 2300.0f);
                    float s = speed(rng);
                    car.phys.vel[0] = dir[0] * s;
                    car.phys.vel[1] = dir[1] * s;
                    car.phys.vel[2] = dir[2] * s;
                }
            }

            std::uniform_real_distribution<float> rand_boost(0.0f, 100.0f);
            car.boost = rand_boost(rng);

            ffi::set_car_state(arena, agent.car_id, car);
        }

        ffi::get_arena_state(arena, current_state);
    }

    [[nodiscard]] std::unique_ptr<StateSetter> clone() const override {
        return std::make_unique<RandomStateSetter>(*this);
    }
};

std::unique_ptr<StateSetter> create_random_state_setter(bool rand_ball_speed, bool rand_car_speed, bool cars_on_ground) {
    return std::make_unique<RandomStateSetter>(rand_ball_speed, rand_car_speed, cars_on_ground);
}

} // namespace buta_ppo::state 