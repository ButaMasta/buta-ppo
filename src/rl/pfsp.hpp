// buta-ppo/src/rl/pfsp.hpp
#pragma once

#include <cstddef>

namespace buta_ppo::rl {

/**
 * @brief Config for Prioritized Fictitious Self-Play: training matches against past versions of the policy.
 *
 * A PFSP env pits the learner's team against a team controlled by a past version. Only the learner's side
 * is trained on. Every update, `versions` past versions are loaded into opponent slots, and each PFSP env
 * plays one slot's version for the whole update. Envs alternate which team the past version controls.
 *
 * NOTE: Only match distributions with even teams get PFSP envs. Uneven layouts are always pure self-play.
 */
struct PFSPConfig {
    // Fraction of each even-team match distribution's envs that play against a past version. 0 disables PFSP.
    float envs = 0.0f;

    // Past versions played each update (opponent slots). Each env group needs at least this many PFSP envs.
    size_t versions = 2;

    [[nodiscard]] bool enabled() const { return envs > 0.0f; }
};

} // namespace buta_ppo::rl
