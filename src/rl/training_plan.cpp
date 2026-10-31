// buta-ppo/src/rl/training_plan.cpp
#include "training_plan.hpp"

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/cuda.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>

namespace buta_ppo::rl {

namespace {
std::string describe_stage(const std::vector<TrainingStage>& stages, size_t stage) {
    std::string text = "stage " + std::to_string(stage + 1) + " of " + std::to_string(stages.size());
    if (!stages[stage].name.empty()) {
        text += " (" + stages[stage].name + ")";
    }
    return text;
}

/**
 * @brief Checks the stage ends, and builds every stage's config to check what it changed.
 */
void validate_plan(const RunnerConfig& base, const std::vector<TrainingStage>& stages) {
    if (stages.empty()) {
        throw std::invalid_argument("A training plan needs at least one stage.");
    }
    for (size_t i = 0; i < stages.size(); i++) {
        const int64_t start = i == 0 ? 0 : stages[i - 1].until_steps;
        if (stages[i].until_steps <= start) {
            throw std::invalid_argument("The training plan's " + describe_stage(stages, i) + " ends at " +
                                        std::to_string(stages[i].until_steps) + " steps, which isn't after " +
                                        (i == 0 ? "0." : "the stage before's end, " + std::to_string(start) + "."));
        }
    }

    RunnerConfig config = base;
    for (size_t i = 0; i < stages.size(); i++) {
        if (stages[i].apply) {
            stages[i].apply(config);
        }
        // Each would make the stage's resume fail or start another bot's run.
        const auto unchanged = [&](bool same, const std::string& setting) {
            if (!same) {
                throw std::invalid_argument("The training plan's " + describe_stage(stages, i) + " changes " + setting +
                                            ", which can't change during a run.");
            }
        };
        unchanged(config.bot_name == base.bot_name, "bot_name");
        unchanged(config.ac_cfg == base.ac_cfg, "ac_cfg");
        unchanged(config.max_players_per_team == base.max_players_per_team, "max_players_per_team");
        unchanged(config.render == base.render && config.benchmark == base.benchmark &&
                  config.calibrate_nexto_ladder == base.calibrate_nexto_ladder,
                  "the mode (render, benchmark or calibrate_nexto_ladder)");
        try {
            config.ppo_cfg.lr_schedule.validate();
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument("The training plan's " + describe_stage(stages, i) + ": " + e.what());
        }
    }
}
} // namespace

RunnerConfig stage_config(const RunnerConfig& base, const std::vector<TrainingStage>& stages, size_t stage) {
    if (stage >= stages.size()) {
        throw std::out_of_range("The training plan has no stage " + std::to_string(stage + 1) + ".");
    }
    RunnerConfig config = base;
    for (size_t i = 0; i <= stage; i++) {
        if (stages[i].apply) {
            stages[i].apply(config);
        }
    }
    config.train_until_steps = stages[stage].until_steps;
    return config;
}

size_t current_stage(const std::vector<TrainingStage>& stages, int64_t steps) {
    for (size_t i = 0; i < stages.size(); i++) {
        if (stages[i].until_steps > steps) return i;
    }
    return stages.size();
}

void run_training_plan(const RunnerConfig& base, const std::vector<TrainingStage>& stages,
                       const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir, int64_t max_steps) {
    if (max_steps < 0) {
        throw std::invalid_argument("A training plan's max_steps must be 0 (train to the plan's end) or more.");
    }
    validate_plan(base, stages);
    int64_t steps = latest_checkpoint_steps(checkpoint_dir, base.bot_name);
    // This call's own end, or 0 for none.
    const int64_t until = max_steps > 0 ? steps + max_steps : 0;

    if (base.render || base.benchmark || base.calibrate_nexto_ladder) {
        const size_t stage = std::min(current_stage(stages, steps), stages.size() - 1);
        std::cout << "Training plan: using the config of " << describe_stage(stages, stage) << ".\n";
        PPORunner runner(stage_config(base, stages, stage));
        runner.run(0, stop_flag, checkpoint_dir);
        return;
    }

    while (!stop_flag) {
        const size_t stage = current_stage(stages, steps);
        if (stage == stages.size()) {
            std::cout << "Training plan complete: the latest checkpoint is at " << steps << " steps, and the last stage "
                      << "ended at " << stages.back().until_steps << ".\n";
            return;
        }
        if (until > 0 && steps >= until) {
            std::cout << "Training plan paused at " << steps << " steps, after the " << max_steps << " steps this run "
                      << "was given, in " << describe_stage(stages, stage) << ". Run it again to carry on.\n";
            return;
        }
        RunnerConfig config = stage_config(base, stages, stage);
        if (until > 0) {
            config.train_until_steps = std::min(config.train_until_steps, until);
        }
        std::cout << "\nTraining plan: " << describe_stage(stages, stage) << ", from " << steps << " to "
                  << config.train_until_steps << " steps"
                  << (config.train_until_steps < stages[stage].until_steps ? " (this run's limit)" : "") << ".\n";

        // Destroyed before the next stage builds its runner, so the two never hold the GPU and env threads at once.
        std::optional<std::string> stop_reason;
        {
            PPORunner runner(config);
            runner.run(std::numeric_limits<int>::max(), stop_flag, checkpoint_dir);
            stop_reason = runner.stop_reason();
        }
        // The old runner's memory stays cached, and the next runner's graph captures can't use any of it (a capture only
        // allocates from its own pool and can't free cached memory), so give it back to the driver first.
        if (torch::cuda::is_available()) {
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
        if (stop_reason) {
            std::cout << "Training plan stopped in " << describe_stage(stages, stage) << " by its stop condition: "
                      << *stop_reason << ".\n";
            return;
        }

        const int64_t new_steps = latest_checkpoint_steps(checkpoint_dir, base.bot_name);
        if (new_steps <= steps && !stop_flag) {
            throw std::runtime_error("The training plan's " + describe_stage(stages, stage) + " ended without a "
                                     "checkpoint past " + std::to_string(steps) + " steps.");
        }
        steps = new_steps;
    }
}

} // namespace buta_ppo::rl
