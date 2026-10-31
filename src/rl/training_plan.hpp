// buta-ppo/src/rl/training_plan.hpp
#pragma once

#include "ppo_runner.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace buta_ppo::rl {

/**
 * @brief One stage of a training plan: the config changes it makes and the lifetime steps it trains up to.
 */
struct TrainingStage {
    // Printed when the stage starts, e.g. "Shot and Save added".
    std::string name;
    // Lifetime steps the stage trains up to. Each stage starts where the one before it ended.
    int64_t until_steps = 0;
    // Changes to the config the stage before trained with. Stage i trains with the plan's base config after stages
    // 0 to i applied their changes, in order. Empty changes nothing.
    std::function<void(RunnerConfig&)> apply;
};

/**
 * @brief The config a stage trains with: the base config after stages 0 to `stage` applied their changes, with
 * `train_until_steps` set to the stage's end.
 *
 * @throws std::out_of_range If there is no such stage.
 */
[[nodiscard]] RunnerConfig stage_config(const RunnerConfig& base, const std::vector<TrainingStage>& stages, size_t stage);

/**
 * @brief The stage a run at `steps` lifetime steps trains in: the first one ending after them.
 *
 * @return The stage's index, or `stages.size()` once every stage is done.
 */
[[nodiscard]] size_t current_stage(const std::vector<TrainingStage>& stages, int64_t steps);

/**
 * @brief Trains through a plan of stages, each with its own config, resuming from the latest checkpoint.
 *
 * Each stage builds a fresh runner from its config, trains up to the stage's end and saves a checkpoint there, then
 * the next stage resumes from that checkpoint, as a restart with the changed config would. On startup the plan picks
 * the stage the latest checkpoint is in, so a stopped or crashed plan carries on where it left off when run again.
 * Every stage's config is built and checked first, so a broken stage fails at startup rather than hours in.
 *
 * `max_steps` limits one call, like `PPORunner::run`'s update count: the stage it ends in trains only to it, and the
 * next call resumes that stage.
 *
 * Render, benchmark and ladder calibration modes run once, with the config of the stage the latest checkpoint is in
 * (the last stage once the plan is done).
 *
 * @param base The config before any stage's changes.
 * @param stages The stages, with increasing `until_steps`.
 * @param stop_flag Stops training after the current update and saves a checkpoint, ending the plan. So does a stage's
 *                  `stop_condition`.
 * @param checkpoint_dir The directory checkpoints are loaded from and saved to.
 * @param max_steps The most steps to train in this call, counted from the latest checkpoint at startup, like the update
 *                  count of `PPORunner::run`: training stops after the update that reaches them and saves a checkpoint,
 *                  and the next call carries on from there, in the middle of a stage if need be. 0 trains to the
 *                  plan's end.
 * @throws std::invalid_argument If `max_steps` is negative, the plan is empty, its ends don't increase, a stage changes a setting that can't
 *         change during a run (`bot_name`, `ac_cfg`, `max_players_per_team`, or the mode), or a stage's
 *         `ppo_cfg.lr_schedule` is invalid.
 */
void run_training_plan(const RunnerConfig& base, const std::vector<TrainingStage>& stages,
                       const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir, int64_t max_steps = 0);

} // namespace buta_ppo::rl
