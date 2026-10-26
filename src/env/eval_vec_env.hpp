// buta-ppo/src/env/eval_vec_env.hpp
#pragma once

#include "buta_ppo/ffi.h"
#include "env/vec_env.hpp"
#include "external/external_bot.hpp"
#include "state/state_setter.hpp"
#include "util/worker_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <torch/torch.h>

// Forward declarations.
namespace buta_ppo::env { class RocketSimEnv; struct StepResult; }

namespace buta_ppo::env {

/**
 * @brief How an evaluation game ended, from the learner's side.
 */
enum class GameOutcome : uint8_t {
    Unfinished,
    Win,
    Loss,
    Draw, // The game cap ran out before a goal.
};

/**
 * @brief One evaluation game: the learner's team against an anchor's team, from a fixed start state.
 *
 * Either team can instead be played by an external bot, whose cars have no rows: the other team by `opponent_bot`
 * instead of an anchor network, and the learner's side by `learner_bot` (e.g. to calibrate one bot against another).
 * Outcomes are always the learner's side's.
 */
struct EvalGame {
    size_t team_size = 1;
    // The anchor network that plays the other team. Unused when `opponent_bot` is set.
    size_t anchor_slot = 0;
    ffi::Team learner_team = ffi::Team::Blue;
    std::shared_ptr<external::ExternalBot> opponent_bot;
    std::shared_ptr<external::ExternalBot> learner_bot;
    // Sets the start state. It should set the same state every time, e.g. `create_eval_kickoff_setter`.
    std::unique_ptr<state::StateSetter> setter;
    // Seeds the env's random number generators (the obs builder's car slot shuffle) before every `reset`, so the game
    // plays out the same each time given the same actions.
    uint32_t seed = 0;
};

/**
 * @brief Runs a fixed set of evaluation games in parallel, each game once from its start state to a goal or the cap.
 *
 * Unlike `VecEnv`, a finished game isn't reset: it stops and keeps its outcome until the next `reset`. Every game
 * keeps its rows in the batched buffers the whole time, so inference always runs on the same batch shape, and the
 * same games with the same actions always give the same results.
 *
 * Learner rows come first, in game order, so they are rows [0, learner agents). Each anchor slot's rows follow, slot
 * by slot, in game order within a slot. Cars of external bots have no rows. Rewards and metrics aren't computed.
 *
 * `reset` plays a range of the games, e.g. the slots that hold an anchor when games are laid out slot by slot. Each
 * game's rows are `game_learner_rows` and `game_anchor_rows`, so a caller can run inference on just the games still
 * running.
 *
 * Games with an external bot play in lockstep (`RocketSimEnv`'s lockstep play), all games a round of a few packets at
 * a time: as many as the bots can wait for their decisions, so twice per step for Nexto. Between rounds, the decisions
 * every game's bots asked for are inferred together, per model and car count, through the bots'
 * `ExternalBatchRunner`s. They give the same games as each env running its bots itself. A range without them steps a
 * whole step at a time, as before.
 */
class EvalVecEnv {
public:
    /**
     * @param games The games to play. Each gets its own env, built once and reused by every `reset`.
     * @param num_anchor_slots The number of anchor networks. Every game's `anchor_slot` must be below it.
     * @param ticks_per_step Physics ticks per step, as in training.
     * @param game_cap_ticks Physics ticks after which a game without a goal ends as a draw.
     * @param max_players_per_team Sizes the observations, as in training.
     * @param num_threads Worker threads stepping the envs.
     * @param worker_cpus The logical CPUs the workers share. Empty leaves them unpinned.
     */
    EvalVecEnv(
        std::vector<EvalGame> games,
        size_t num_anchor_slots,
        int ticks_per_step,
        int game_cap_ticks,
        size_t max_players_per_team,
        size_t num_threads,
        const std::vector<int>& worker_cpus = {}
    );

    ~EvalVecEnv();

    /**
     * @brief Puts the games in [begin, end) back at their start state, with their envs' random number generators
     * reseeded, and external bots given fresh controllers. Only they are played until the next `reset`; the others
     * aren't stepped and stay `Unfinished`.
     *
     * Clears the outcomes and fills those games' rows of the batched obs and action masks. External bots play their
     * frozen packets before the first step (the join window and the kickoff countdown) here.
     */
    void reset(size_t begin, size_t end);

    /**
     * @brief Plays the first `num_games` games, in [1, game count]. All of them by default.
     */
    void reset(size_t num_games = SIZE_MAX) { reset(0, num_games); }

    /**
     * @brief Steps every unfinished game being played once. Finished games are skipped and keep their rows unchanged.
     *
     * @param batched_actions An action for every row of the games being played. Other rows are ignored.
     */
    void step(const int* batched_actions);

    /**
     * @brief Hands a game's external team (its learner's side or its opponent, whichever was given a bot) to another
     * bot of the same model, e.g. Nexto with another beta. It plays from the next `reset`.
     */
    void set_external_bot(size_t game, bool learner_side, std::shared_ptr<external::ExternalBot> bot);

    [[nodiscard]] bool all_done() const { return games_left_ == 0; }
    [[nodiscard]] size_t games_left() const { return games_left_; }
    // The games the last `reset` started: [played_begin(), played_end()).
    [[nodiscard]] size_t played_begin() const { return played_begin_; }
    [[nodiscard]] size_t played_end() const { return played_end_; }
    // The end of the played range, which is the number played when it starts at 0.
    [[nodiscard]] size_t played_games() const { return played_end_; }

    // Each game's outcome, in the order the games were given.
    [[nodiscard]] const std::vector<GameOutcome>& outcomes() const { return outcomes_; }
    // Each game's length in physics ticks, up to its goal or the cap.
    [[nodiscard]] const std::vector<int>& game_ticks() const { return game_ticks_; }

    [[nodiscard]] RowRange learner_rows() const { return {0, learner_agents_}; }
    // A game's learner rows and its anchor rows, contiguous, one per car of the team, or none for an external bot's team.
    [[nodiscard]] RowRange game_learner_rows(size_t game) const {
        return {learner_rows_before_[game], learner_rows_before_[game + 1] - learner_rows_before_[game]};
    }
    [[nodiscard]] RowRange game_anchor_rows(size_t game) const { return {anchor_row_begin_[game], anchor_row_count_[game]}; }
    [[nodiscard]] RowRange anchor_rows(size_t slot) const { return anchor_rows_.at(slot); }

    // Batched CPU buffers (pinned), one row per agent. Their addresses never change.
    [[nodiscard]] const torch::Tensor& obs() const { return batched_obs_; }
    [[nodiscard]] const torch::Tensor& action_masks() const { return batched_action_masks_; }

    [[nodiscard]] size_t get_num_games() const { return envs_.size(); }
    // Every agent's row, which is the row count of the batched buffers and the batched actions.
    [[nodiscard]] size_t get_total_agents() const { return total_agents_; }
    [[nodiscard]] size_t get_single_obs_size() const { return single_obs_size_; }
    [[nodiscard]] size_t get_action_space_size() const { return action_space_size_; }

private:
    std::vector<std::unique_ptr<RocketSimEnv>> envs_;
    std::vector<ffi::Team> learner_teams_;
    std::vector<uint32_t> seeds_;

    size_t single_obs_size_ = 0;
    size_t action_space_size_ = 0;
    size_t total_agents_ = 0;
    size_t learner_agents_ = 0;
    std::vector<RowRange> anchor_rows_;
    // The learner rows of the games before each game, and of all games at the end, since they're in game order.
    std::vector<size_t> learner_rows_before_;
    // Each game's first anchor row and its anchor rows.
    std::vector<size_t> anchor_row_begin_;
    std::vector<size_t> anchor_row_count_;

    // Batched row of each env's agents, in env order, or `NO_ROW` for an external bot's car. Env i's agents start at
    // `env_agent_offsets_[i]`.
    static constexpr uint32_t NO_ROW = UINT32_MAX;
    std::vector<uint32_t> agent_rows_;
    std::vector<size_t> env_agent_offsets_;
    std::vector<size_t> env_agent_counts_;
    // Per-env scratch for gathering an env's actions from its rows. Indexed like `agent_rows_`.
    std::vector<int> env_actions_;

    // Written only by the worker stepping the env, read by the main thread between batches.
    std::vector<GameOutcome> outcomes_;
    std::vector<int> game_ticks_;
    size_t games_left_ = 0;
    size_t played_begin_ = 0;
    size_t played_end_ = 0;

    int ticks_per_step_ = 0;

    // Lockstep play of the games with external bots.
    std::vector<uint8_t> external_;        // Per env: whether it has an external bot.
    bool range_external_ = false;          // Whether the played range has a game with one.
    // Per env, written by the worker stepping it: frozen packets left before the first step, whether its step has
    // ended, and whether it has begun a packet whose decisions aren't inferred yet.
    std::vector<int> frozen_left_;
    std::vector<uint8_t> step_done_;
    std::vector<uint8_t> needs_inference_;
    int lockstep_tick_ = 0;                // The step's ticks played before this round, set before each batch.
    int round_ticks_ = 1;                  // The ticks this round plays.
    int window_ = 1;                       // The packets a round plays: the played games' shortest lockstep window.
    std::vector<std::vector<size_t>> sized_decisions_; // Per env, per external group: the decisions per round its slot was sized for.

    // Batched external inference: a runner per model, and a slot per model and car count gathering the obs of every
    // game's cars of that kind deciding on the current packet.
    struct InferenceSlot {
        external::ExternalBatchRunner* runner = nullptr;
        size_t num_cars = 0;
        size_t obs_size = 0;
        size_t decision_size = 0;
        size_t count = 0; // Gathered this packet.
    };
    std::vector<std::unique_ptr<external::ExternalBatchRunner>> runners_;
    std::vector<InferenceSlot> slots_;
    std::vector<std::vector<size_t>> group_slots_; // Per env, per external group: its slot.
    std::vector<std::vector<const void*>> group_models_; // Per env, per external group: its bot's model.
    std::vector<uint8_t> has_learner_bot_;
    // This packet's gathered batches, in gathering order, so each slot's decisions go back where they came from.
    struct Gathered {
        size_t env;
        size_t group;
        size_t slot;
        size_t offset; // In the slot's batch.
    };
    std::vector<Gathered> gathered_;
    std::vector<const float*> slot_decisions_; // Each slot's decisions, once its launch is done.

    torch::Tensor batched_obs_;
    torch::Tensor batched_action_masks_;

    // The batched actions of the step being run. Written by the main thread before each step batch.
    const int* current_actions_ptr_ = nullptr;

    // The workers' tasks, built once so dispatching a batch doesn't allocate.
    util::WorkerPool::Task reset_task_;
    util::WorkerPool::Task step_task_;
    util::WorkerPool::Task lockstep_reset_task_;
    util::WorkerPool::Task frozen_task_;
    util::WorkerPool::Task tick_task_;

    // Last, so the workers are stopped before anything they use is destroyed.
    util::WorkerPool pool_;

    /**
     * @brief Steps one unfinished game with its actions from the current batched actions and records how it ended.
     */
    void step_env(size_t env_idx);

    /**
     * @brief Gathers an env's actions from its rows of the current batched actions. External cars get 0, unused.
     */
    const int* gather_actions(size_t env_idx);

    /**
     * @brief Records a step's result: the game's ticks, and its outcome or its new obs.
     */
    void record_step(size_t env_idx, const StepResult& result);

    /**
     * @brief Runs the external bots' inference for every game in the played range that has begun a packet: gathers
     * their obs per slot, launches every slot, then hands each game its decisions.
     */
    void infer_external();

    /**
     * @brief Scatters an env's per-agent obs and action masks into its agents' rows of the batched buffers.
     */
    void write_obs_and_masks(size_t env_idx, const std::vector<float>& obs, const std::vector<float>& masks);
};

} // namespace buta_ppo::env
