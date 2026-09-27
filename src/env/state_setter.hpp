// buta-ppo/src/env/state_setter.hpp
#pragma once

#include "buta_ppo/ffi.h"

#include <vector>
#include <random>
#include <memory>
#include <type_traits>
#include <utility>

// Forward declaration.
namespace buta_ppo::env { struct AgentMeta; }

namespace buta_ppo::env {

/**
 * @brief Abstract base class to be extended ensuring state setters have the necessary methods.
 */
class StateSetter {
public:
    virtual ~StateSetter() = default;

    /**
     * @brief Applies the state setters configuration to the supplied arena.
     * 
     * @param arena The arena to set the state for.
     * @param current_state The current state buffer to fill when the state setter is complete.
     * @param agents The list of agents in the arena.
     * @param rng The random device to use.
     */
    virtual void apply(
        const ffi::ArenaPtr& arena,
        ffi::ArenaState& current_state,
        const std::vector<AgentMeta>& agents,
        std::mt19937& rng
    ) = 0;

    [[nodiscard]] virtual std::unique_ptr<StateSetter> clone() const = 0;
};

/**
 * @brief The state setter to represent normal rocket league random kickoff selection.
 */
class DefaultKickoffSetter : public StateSetter {
public:
    void apply(
        const ffi::ArenaPtr& arena, 
        ffi::ArenaState& current_state, 
        const std::vector<AgentMeta>& agents, 
        std::mt19937& rng
    ) override;

    [[nodiscard]] std::unique_ptr<StateSetter> clone() const override;
};

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
        const std::vector<AgentMeta>& agents, 
        std::mt19937& rng
    ) override;

    [[nodiscard]] std::unique_ptr<StateSetter> clone() const override;
};

/**
 * @brief Allows for a user friendly way to define a state setter with a desired weight to appear.
 */
struct StateSetterDistribution {
    std::unique_ptr<StateSetter> setter;
    float weight = 1.0f;

    StateSetterDistribution(std::unique_ptr<StateSetter> s, float w)
        : setter(std::move(s)), weight(w) {}

    template <typename T, typename = std::enable_if_t<std::is_base_of_v<StateSetter, std::decay_t<T>>>>
    StateSetterDistribution(T&& instance, float w)
        : setter(std::make_unique<std::decay_t<T>>(std::forward<T>(instance))), weight(w) {}

    StateSetterDistribution(const StateSetterDistribution& other)
        : setter(other.setter->clone()), weight(other.weight) {}
        
    StateSetterDistribution& operator=(const StateSetterDistribution& other) {
        if (this != &other) {
            setter = other.setter->clone();
            weight = other.weight;
        }
        return *this;
    }
};

} // namespace buta_ppo::env