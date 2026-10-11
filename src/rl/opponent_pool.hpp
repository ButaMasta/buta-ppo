// buta-ppo/src/rl/opponent_pool.hpp
#pragma once

#include "actor_critic.hpp"
#include "pfsp.hpp"

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace buta_ppo::rl {

/**
 * @brief The pool of past learner versions that PFSP samples opponents from, with the learner's win rate against each.
 *
 * Once opened on a directory, the pool persists there: each snapshot as a full model file `<steps>.pt` (loadable
 * like any checkpoint's `model.pt`), the win records in `state.txt`, and the network shape in `meta.txt`.
 * Without a directory it lives only in memory.
 */
class OpponentPool {
public:
    struct Snapshot {
        // Lifetime steps of the learner when the snapshot was taken. Identifies the version.
        int64_t steps = 0;

        // Pinned CPU copies of the weights in `parameters()` order. Critic weights are left undefined, since
        // opponents never run the critic.
        std::vector<torch::Tensor> params;

        // The learner's decayed results against this snapshot.
        double wins = 0.0;
        double games = 0.0;

        /**
         * @brief The learner's estimated win rate against this snapshot, starting at 0.5 with no games.
         */
        [[nodiscard]] double win_rate() const { return (wins + 1.0) / (games + 2.0); }
    };

    /**
     * @param config The PFSP config.
     * @param ac_config The network config of the learner, with its obs and action sizes filled in.
     * @param seed Seed for sampling.
     */
    OpponentPool(const PFSPConfig& config, const ActorCriticConfig& ac_config, uint32_t seed = std::random_device{}());

    /**
     * @brief Loads the pool saved in `dir`, and keeps the pool saved there from then on.
     *
     * Snapshots newer than `max_steps` come from a run that was rolled back to an older checkpoint, so their files
     * are deleted. A pool saved with a different network shape can't be played, so its directory is renamed to
     * `<dir>.stale-<unix time>` and a fresh pool is started.
     *
     * @param dir The pool's directory. Created if missing.
     * @param max_steps The lifetime steps of the checkpoint being resumed.
     */
    void open(const std::filesystem::path& dir, int64_t max_steps);

    /**
     * @brief Writes the win records to the pool's directory. Does nothing for an in-memory pool.
     */
    void save_state() const;

    /**
     * @brief Adds a snapshot of the learner's current weights, evicting one if the pool is over its size.
     *
     * Saves the snapshot and the win records when the pool has a directory. Invalidates the indices from `sample`.
     *
     * @param steps The learner's lifetime steps.
     * @param learner The learner to copy.
     */
    void add(int64_t steps, const ActorCritic& learner);

    /**
     * @brief Samples snapshots to play by their PFSP weights. Distinct while the pool has enough snapshots.
     *
     * @param count The number of snapshots to sample.
     * @return Indices into the pool, valid until the next `add`.
     */
    [[nodiscard]] std::vector<size_t> sample(size_t count);

    /**
     * @brief Records the learner's results against a snapshot. Each game decays the snapshot's earlier results.
     *
     * @param idx The snapshot's index from `sample`.
     * @param wins The learner's wins.
     * @param losses The learner's losses.
     */
    void record(size_t idx, uint32_t wins, uint32_t losses);

    /**
     * @brief The PFSP sampling weight of a snapshot, in [min_weight, 1].
     */
    [[nodiscard]] double weight(const Snapshot& snapshot) const;

    [[nodiscard]] const Snapshot& operator[](size_t idx) const { return snapshots_.at(idx); }
    [[nodiscard]] size_t size() const { return snapshots_.size(); }
    [[nodiscard]] bool empty() const { return snapshots_.empty(); }

private:
    // Snapshots with fewer decayed games than this aren't trusted enough to be evicted for being beaten.
    static constexpr double EVICT_MIN_GAMES = 20.0;

    PFSPConfig config_;
    ActorCriticConfig ac_config_;
    std::vector<Snapshot> snapshots_; // Oldest first.
    std::mt19937 rng_;
    std::filesystem::path dir_; // Empty for an in-memory pool.

    /**
     * @brief Copies a network's non-critic weights into a new snapshot.
     */
    [[nodiscard]] static Snapshot make_snapshot(int64_t steps, const ActorCritic& net);

    /**
     * @brief The network shape the snapshots must match, as written to `meta.txt`.
     */
    [[nodiscard]] std::string meta_text() const;

    /**
     * @brief Removes the snapshot the learner beats most among those with enough games, or else the oldest.
     * The newest snapshot is never removed. Deletes its file when the pool has a directory.
     */
    void evict();
};

} // namespace buta_ppo::rl
