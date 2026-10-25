// buta-ppo/src/env/eval_vec_env.hpp
#pragma once

#include "buta_ppo/ffi.h"
#include "env/vec_env.hpp"
#include "state/state_setter.hpp"
#include "util/worker_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <torch/torch.h>

// Forward declarations.
namespace buta_ppo::env { class RocketSimEnv; }

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
 */
struct EvalGame {
    size_t team_size = 1;
    // The anchor network that plays the other team.
    size_t anchor_slot = 0;
    ffi::Team learner_team = ffi::Team::Blue;
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
 * by slot, in game order within a slot. Rewards and metrics aren't computed.
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
     * @brief Puts every game back at its start state, with its env's random number generators reseeded.
     *
     * Clears the outcomes and fills the batched obs and action masks.
     */
    void reset();

    /**
     * @brief Steps every unfinished game once. Finished games are skipped and keep their rows unchanged.
     *
     * @param batched_actions An action for every row of the batched buffers. Finished games' rows are ignored.
     */
    void step(const int* batched_actions);

    [[nodiscard]] bool all_done() const { return games_left_ == 0; }
    [[nodiscard]] size_t games_left() const { return games_left_; }

    // Each game's outcome, in the order the games were given.
    [[nodiscard]] const std::vector<GameOutcome>& outcomes() const { return outcomes_; }
    // Each game's length in physics ticks, up to its goal or the cap.
    [[nodiscard]] const std::vector<int>& game_ticks() const { return game_ticks_; }

    [[nodiscard]] RowRange learner_rows() const { return {0, learner_agents_}; }
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

    // Batched row of each env's agents, in env order. Env i's agents start at `env_agent_offsets_[i]`.
    std::vector<uint32_t> agent_rows_;
    std::vector<size_t> env_agent_offsets_;
    std::vector<size_t> env_agent_counts_;
    // Per-env scratch for gathering an env's actions from its rows. Indexed like `agent_rows_`.
    std::vector<int> env_actions_;

    // Written only by the worker stepping the env, read by the main thread between batches.
    std::vector<GameOutcome> outcomes_;
    std::vector<int> game_ticks_;
    size_t games_left_ = 0;

    torch::Tensor batched_obs_;
    torch::Tensor batched_action_masks_;

    // The batched actions of the step being run. Written by the main thread before each step batch.
    const int* current_actions_ptr_ = nullptr;

    // The workers' tasks, built once so dispatching a batch doesn't allocate.
    util::WorkerPool::Task reset_task_;
    util::WorkerPool::Task step_task_;

    // Last, so the workers are stopped before anything they use is destroyed.
    util::WorkerPool pool_;

    /**
     * @brief Steps one unfinished game with its actions from the current batched actions and records how it ended.
     */
    void step_env(size_t env_idx);

    /**
     * @brief Scatters an env's per-agent obs and action masks into its agents' rows of the batched buffers.
     */
    void write_obs_and_masks(size_t env_idx, const std::vector<float>& obs, const std::vector<float>& masks);
};

} // namespace buta_ppo::env
