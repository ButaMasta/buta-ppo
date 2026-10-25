// buta-ppo/src/state/eval_kickoff.cpp
#include "setters.hpp"
#include "state_setter.hpp"
#include "util/math_utils.hpp"
#include "env/rocketsim_env.hpp"

#include <memory>
#include <random>
#include <cstdint>

namespace buta_ppo::state {

/**
 * @brief The state setter to represent normal rocket league random kickoff 
 * selection with small perturbations for evaluations of discrete policies.
 */
class EvalKickoffSetter : public StateSetter {
private:
    uint32_t bank_seed;
    uint32_t index;
    float pos_jitter;
    float yaw_jitter;

public:
    EvalKickoffSetter(uint32_t bank_seed, uint32_t index, float pos_jitter, float yaw_jitter) 
    : bank_seed(bank_seed), index(index), pos_jitter(pos_jitter), yaw_jitter(yaw_jitter) {}

    void apply(
        const ffi::ArenaPtr& arena, 
        ffi::ArenaState& current_state, 
        const std::vector<env::AgentMeta>& agents, 
        std::mt19937& /*rng*/
    ) override {
        std::seed_seq seq{bank_seed, index};
        std::mt19937 local_rng(seq);
        ffi::reset_to_random_kickoff(arena, local_rng(), true);
        ffi::get_arena_state(arena, current_state);
        
        for (const env::AgentMeta& agent : agents) {
            auto& agent_car = current_state.cars[agent.car_id];
            auto& agent_car_phys = agent_car.phys;
            std::uniform_real_distribution<float> pos_jitter_rand(-pos_jitter, pos_jitter);
            std::uniform_real_distribution<float> yaw_jitter_rand(-yaw_jitter, yaw_jitter);
            
            agent_car_phys.pos[0] += pos_jitter_rand(local_rng);
            agent_car_phys.pos[1] += pos_jitter_rand(local_rng);
            
            float pitch, yaw, roll;
            util::math::mat3_to_euler(agent_car_phys.rot_mat, pitch, yaw, roll);
            yaw += yaw_jitter_rand(local_rng);
            util::math::euler_to_mat3(pitch, yaw, roll, agent_car_phys.rot_mat);
            
            ffi::set_car_state(arena, agent.car_id, agent_car);
        }

        ffi::get_arena_state(arena, current_state);
    }

    [[nodiscard]] std::unique_ptr<StateSetter> clone() const override {
        return std::make_unique<EvalKickoffSetter>(*this);
    }
};

std::unique_ptr<StateSetter> create_eval_kickoff_setter(uint32_t bank_seed, uint32_t index, float pos_jitter, float yaw_jitter) {
    return std::make_unique<EvalKickoffSetter>(bank_seed, index, pos_jitter, yaw_jitter);
}

} // namespace buta_ppo::state 