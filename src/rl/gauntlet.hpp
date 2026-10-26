// buta-ppo/src/rl/gauntlet.hpp
#pragma once

#include "actor_critic.hpp"
#include "eval_ledger.hpp"
#include "env/eval_vec_env.hpp"

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace buta_ppo::rl {

/**
 * @brief Config for evaluation gauntlets: periodic matches of the learner against fixed anchors, rated with
 * Bradley–Terry, to track its skill over training.
 *
 * Every `steps_per_eval` lifetime steps, training pauses and the learner plays every active anchor from a fixed bank
 * of jittered kickoffs, each start state once from each side, with both sides playing their most likely action.
 * Results go to `<checkpoint dir>/<bot name>_eval/`.
 */
struct EvalConfig {
    // Lifetime steps between evaluations. 0 turns evaluation off.
    int64_t steps_per_eval = 50'000'000;

    // Team sizes rated, each played as NvN with its own rating. Empty uses every even layout in the match distributions.
    std::vector<size_t> team_sizes = {};

    // Start states per team size. Each is played from both sides against every active anchor.
    size_t bank_size = 64;
    uint32_t bank_seed = 12345;
    float pos_jitter = 2.5f;  // Kickoff car position jitter, in uu.
    float yaw_jitter = 0.01f; // Kickoff car yaw jitter, in radians.

    // A game with no goal by then is a draw.
    float game_cap_seconds = 150.0f;

    // Most anchors the learner plays at once. When more qualify, the newest are played.
    size_t active_anchors = 4;
    // The learner becomes an anchor once its score (draws count half) against the newest anchor, over all team sizes,
    // reaches this.
    float promote_win_rate = 0.7f;
    // An anchor isn't played while the learner's predicted score against it, from their ratings and averaged over the
    // team sizes, is above this: those games say little about the learner. It's played again if the learner falls
    // back. The newest anchor is always played. 1 plays every anchor (up to `active_anchors`).
    float retire_win_rate = 0.85f;

    // Draws added against every anchor when fitting a rating, which keeps it finite after a perfect record.
    int virtual_draws = 1;

    [[nodiscard]] bool enabled() const { return steps_per_eval > 0; }
};

/**
 * @brief The learner's results in one team size from one evaluation.
 */
struct EvalSizeResult {
    size_t team_size = 0;
    double elo = 0.0;
    double elo_std_error = 0.0;
    int wins = 0;
    int losses = 0;
    int draws = 0;
    double mean_game_seconds = 0.0;
};

/**
 * @brief The results of one `Gauntlet::run`, for logging.
 */
struct EvalResult {
    bool played = false;   // False when there was no anchor yet, so the learner only became the first one.
    bool promoted = false; // Whether the learner became an anchor.
    std::vector<int64_t> anchors_played; // Lifetime steps of the anchors played, oldest first.
    std::vector<EvalSizeResult> sizes;
    double seconds = 0.0;
};

/**
 * @brief Runs evaluation gauntlets and keeps their anchors and results.
 *
 * Anchors are past learners, saved as `snapshot_<steps>.pt` model files next to the ledger. The first one is the
 * reference, rated 0 in every team size; every later anchor's ratings are the learner's fitted ratings at the
 * evaluation that promoted it, and never change.
 */
class Gauntlet {
public:
    /**
     * @param config The evaluation settings.
     * @param model_config The learner's network and obs settings, which the anchors share.
     * @param team_sizes The team sizes to rate, already resolved from the config.
     * @param num_threads Worker threads stepping the evaluation envs.
     * @param worker_cpus The logical CPUs the workers share. Empty leaves them unpinned.
     * @param device Where the learner and the anchors run.
     */
    Gauntlet(const EvalConfig& config, const ModelConfig& model_config, std::vector<size_t> team_sizes,
             size_t num_threads, const std::vector<int>& worker_cpus, torch::Device device);

    /**
     * @brief Loads the anchors and ledger in `dir`, or starts fresh there. Anything newer than `max_steps` belongs to
     * a rolled-back run and is deleted.
     */
    void open(const std::filesystem::path& dir, int64_t max_steps);

    /**
     * @brief Plays the learner against the active anchors, rates it, and promotes it if it passed the newest anchor.
     * With no anchors yet, the learner only becomes the first one.
     */
    EvalResult run(int64_t learner_steps, ActorCritic learner);

    // Whether there's no anchor yet, so the next `run` only makes the learner the first one.
    [[nodiscard]] bool empty() const { return anchors_.empty(); }

private:
    EvalConfig config_;
    ModelConfig model_config_;
    std::vector<size_t> team_sizes_;
    torch::Device device_;

    // Games are laid out slot by slot, then team size, then bank state, then side (learner blue first). Only the slots
    // holding an active anchor are played, which are the first ones, so their games are a prefix.
    std::unique_ptr<env::EvalVecEnv> env_;
    // One network per active anchor.
    std::vector<ActorCritic> slots_;
    // The anchor whose weights each slot holds, or -1 for none, so unchanged slots aren't reloaded.
    std::vector<int64_t> slot_anchors_;
    // Every anchor's lifetime steps, oldest first.
    std::vector<int64_t> anchors_;
    // The anchors the next games are played against, oldest first, so slot i holds active_[i] and the newest anchor
    // is in the last one. Chosen by `select_active`.
    std::vector<int64_t> active_;

    EvalLedger ledger_;
    std::filesystem::path dir_;

    // Compact inference buffers, sized for every row of the evaluation envs. Each step, the rows of the games still
    // running are gathered to the front: the learner's rows, then each slot's.
    std::vector<size_t> row_index_; // The batched row each compact row comes from.
    torch::Tensor obs_staging_;     // Pinned CPU: the gathered obs and masks, uploaded from here.
    torch::Tensor masks_staging_;
    torch::Tensor obs_gpu_;
    torch::Tensor masks_gpu_;
    torch::Tensor actions_gpu_;
    torch::Tensor actions_cpu_;     // Pinned CPU: the compact actions, downloaded here.
    std::vector<int> step_actions_; // Every batched row's action for `EvalVecEnv::step`, scattered from the compact ones.

    // An anchor's name in the ledger, e.g. "snapshot_465862848", and its model file next to the ledger.
    [[nodiscard]] static std::string anchor_id(int64_t steps);
    [[nodiscard]] std::filesystem::path anchor_path(int64_t steps) const;
    // The settings that make ratings comparable, saved as meta.txt. A change starts a fresh ledger.
    [[nodiscard]] std::string meta_text() const;
    // The learner's predicted score against an anchor, σ(learner's latest rating − anchor's rating), averaged over the
    // team sizes where both are rated. None when no size has both.
    [[nodiscard]] std::optional<double> predicted_score(int64_t anchor) const;
    // Chooses the anchors to play, newest first: the newest always, then every anchor whose predicted score is at
    // most `retire_win_rate` or unknown, up to `active_anchors`. Then loads them into the slots.
    void select_active();
    // Saves the learner as an anchor.
    void promote(int64_t steps, const ActorCritic& learner);

};

} // namespace buta_ppo::rl
