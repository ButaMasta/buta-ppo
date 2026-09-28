// buta-ppo/src/state/default_kickoff.cpp
#include "setters.hpp"
#include "state_setter.hpp"

#include <memory>

namespace buta_ppo::state {

/**
 * @brief The state setter to represent normal rocket league random kickoff selection.
 */
class DefaultKickoffSetter : public StateSetter {
public:
    void apply(
        const ffi::ArenaPtr& arena, 
        ffi::ArenaState& current_state, 
        const std::vector<env::AgentMeta>& /*agents*/, 
        std::mt19937& rng
    ) override {
        ffi::reset_to_random_kickoff(arena, rng(), true);
        ffi::get_arena_state(arena, current_state);
    }

    [[nodiscard]] std::unique_ptr<StateSetter> clone() const override {
        return std::make_unique<DefaultKickoffSetter>(*this);
    }
};

std::unique_ptr<StateSetter> create_default_kickoff_setter() {
    return std::make_unique<DefaultKickoffSetter>();
}

} // namespace buta_ppo::state 