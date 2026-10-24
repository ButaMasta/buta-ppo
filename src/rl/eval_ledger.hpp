// buta-ppo/src/rl/eval_ledger.hpp
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace buta_ppo::rl {

/**
 * @brief The record of evaluation games and the learner ratings fitted from them.
 *
 * Persists in its directory as `results.txt` (the learner's results against each anchor at each evaluation),
 * `ratings.txt` (the learner's fitted rating at each evaluation) and `meta.txt` (the evaluation settings).
 * Past rows never change. Only rows newer than a resumed checkpoint are dropped.
 */
class EvalLedger {
public:
    struct Result {
        int64_t learner_steps = 0;
        int team_size = 0;
        std::string opponent; // opponent name: "snapshot_50000000" or "nexto_beta_1.0"
        int wins = 0;
        int losses = 0;
        int draws = 0;
    };

    struct Rating {
        int64_t learner_steps = 0;
        int team_size = 0;
        double rating = 0.0; // BT units, not Elo
        double std_error = 0.0;
    };

    // Loads the ledger in `dir`, or starts a fresh one there. A ledger saved with a different `meta` is moved aside
    // to `<dir>.stale-<time>`. Rows newer than `max_steps` belong to a rolled-back run and are dropped.
    void open(const std::filesystem::path& dir, const std::string& meta, int64_t max_steps);

    void add_result(const Result& result);
    void add_rating(const Rating& rating);

    // The learner's rating at `learner_steps` in `team_size`, if one was recorded.
    [[nodiscard]] std::optional<Rating> find_rating(int64_t learner_steps, int team_size) const;

    [[nodiscard]] const std::vector<Result>& results() const { return results_; }
    [[nodiscard]] const std::vector<Rating>& ratings() const { return ratings_; }

    // Rewrites `results.txt` and `ratings.txt`.
    void save() const;

private:
    std::filesystem::path dir_;
    std::vector<Result> results_;
    std::vector<Rating> ratings_;
};

} // namespace buta_ppo::rl
