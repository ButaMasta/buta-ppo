// buta-ppo/src/rl/pfsp.hpp
#pragma once

#include <cstddef>
#include <cstdint>

namespace buta_ppo::rl {

/**
 * @brief How strongly PFSP favors a past version given the learner's estimated win rate p against it.
 */
enum class PFSPWeighting {
    Variance, // 4p(1 - p): favors evenly matched versions. Suits early training, when most versions are similar.
    Hard      // (1 - p)^hard_exponent: favors the versions the learner still loses to.
};

/**
 * @brief Config for Prioritized Fictitious Self-Play: training matches against past versions of the policy.
 *
 * A PFSP env pits the learner's team against a team controlled by a past version. Only the learner's side
 * is trained on. Every update, `versions` past versions are sampled from the snapshot pool and loaded into
 * opponent slots, and each PFSP env plays one slot's version for the whole update. Envs alternate which team
 * the past version controls.
 *
 * NOTE: Only match distributions with even teams get PFSP envs. Uneven layouts are always pure self-play.
 * NOTE: `envs = 0` disables PFSP entirely, with no cost to the rollout.
 */
struct PFSPConfig {
    // Fraction of each even-team match distribution's envs that play against a past version. 0 disables PFSP.
    float envs = 0.0f;

    // Past versions played each update (opponent slots). Each env group needs at least this many PFSP envs.
    size_t versions = 2;

    // Until this many lifetime steps, PFSP envs play a copy of the current learner instead of past versions.
    // Snapshots are still added to the pool in the meantime.
    int64_t start_steps = 0;

    // Lifetime steps between snapshots of the learner added to the pool.
    int64_t snapshot_interval = 5'000'000;

    // Max snapshots kept. Past it, the snapshot the learner beats most is evicted and its file deleted
    // (the newest is always kept). The pool is saved in `<checkpoint dir>/<bot name>_pool/`.
    size_t pool_size = 32;

    PFSPWeighting weighting = PFSPWeighting::Variance;
    float hard_exponent = 2.0f;

    // Floor on a snapshot's sampling weight (weights are in [0, 1]), so no snapshot is ignored completely.
    float min_weight = 0.02f;

    // Approximate number of most recent games a snapshot's win rate is estimated over. Older games decay away,
    // since the learner keeps improving.
    float win_rate_window = 50.0f;

    [[nodiscard]] bool enabled() const { return envs > 0.0f; }
};

} // namespace buta_ppo::rl
