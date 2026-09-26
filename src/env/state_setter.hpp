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

class StateSetter {
public:
    virtual ~StateSetter() = default;

    virtual void apply(
        const ffi::ArenaPtr& arena,
        ffi::ArenaState& current_state,
        [[maybe_unused]] const std::vector<AgentMeta>& agents,
        std::mt19937& rng
    ) = 0;

    [[nodiscard]] virtual std::unique_ptr<StateSetter> clone() const = 0;
};

class DefaultKickoffSetter : public StateSetter {
public:
    void apply(const ffi::ArenaPtr& arena, ffi::ArenaState& current_state, [[maybe_unused]] const std::vector<AgentMeta>& agents, std::mt19937& rng) override;

    [[nodiscard]] std::unique_ptr<StateSetter> clone() const override;
};

class RandomStateSetter : public StateSetter {
private:
    bool rand_ball_speed_;
    bool rand_car_speed_;
    bool cars_on_ground_;

public:
    RandomStateSetter(bool rand_ball_speed, bool rand_car_speed, bool cars_on_ground)
        : rand_ball_speed_(rand_ball_speed), rand_car_speed_(rand_car_speed), cars_on_ground_(cars_on_ground) {}
    
    void apply(const ffi::ArenaPtr& arena, ffi::ArenaState& current_state, const std::vector<AgentMeta>& agents, std::mt19937& rng) override;

    [[nodiscard]] std::unique_ptr<StateSetter> clone() const override;
};

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

}; // namespace buta_ppo::env