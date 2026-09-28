// buta-ppo/src/env/state_setter.hpp
#pragma once

#include "buta_ppo/ffi.h"

#include <vector>
#include <random>
#include <memory>
#include <utility>

// Forward declaration.
namespace buta_ppo::env { struct AgentMeta; }

namespace buta_ppo::state {

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
        const std::vector<env::AgentMeta>& agents,
        std::mt19937& rng
    ) = 0;

    [[nodiscard]] virtual std::unique_ptr<StateSetter> clone() const = 0;
};

/**
 * @brief Allows for a user friendly way to define a state setter with a desired weight to appear.
 */
struct StateSetterDistribution {
    std::unique_ptr<StateSetter> setter;
    float weight = 1.0f;

    StateSetterDistribution(std::unique_ptr<StateSetter> s, float w)
        : setter(std::move(s)), weight(w) {}

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

} // namespace buta_ppo::state