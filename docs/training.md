# Training

On startup, the framework searches `checkpoints/` for the most recent checkpoint matching the configured `bot_name`. If one is found, training resumes from its step count. Otherwise, a fresh model is initialized. A checkpoint has to match the configured network: if the `ac_cfg` layers or layer norm changed, or the obs size did (through `max_players_per_team` or the obs builder), startup fails with an error showing the settings the checkpoint was trained with next to the configured ones. Change `bot_name` or move the old checkpoints out of `checkpoints/` to start fresh. A changed `ticks_per_step` only prints a `[WARN]`, and training continues with the new value.

A run trains for at most the number of updates passed to `runner.run` in `main.cpp` (`1'000` by default, about 100M steps with the demo config), then saves a checkpoint and exits. Run the binary again to continue from there. To stop at a step count instead, set `config.train_until_steps`: training stops after the update that reaches it and saves a checkpoint, and a start at or past it trains nothing (`0`, the default, has no limit).

To stop training safely, press Ctrl+C, or send `SIGTERM` (what `kill`, `systemctl stop`, and `docker stop` send). The current update finishes, a checkpoint is saved, and the program exits.

Checkpoints are also saved every `config.steps_per_checkpoint` lifetime steps (`100'000'000` by default; `0` turns this off), so a crash only loses the progress since the last one. Every checkpoint is kept, and each one is roughly as large as the network and its optimizer state (about 50 MB for the demo network), so delete old ones you don't need.

## Training Plans

A plan trains through stages, each changing the config and training up to a step count, so a run can follow a curriculum unattended: add rewards in phases, lower the learning rate, shift the match or state setter weights. Build the base config as usual, then run the stages instead of `runner.run`:

```cpp
#include "rl/training_plan.hpp"

const std::vector<TrainingStage> plan = {
    {"Learn to play", 1'000'000'000, {}},
    {"Shot and Save", 3'000'000'000, [](RunnerConfig& c) {
        c.reward_entries.push_back({"Shot", create_shot_reward(4.0f), 1.5f});
    }},
    {"Decay lr", 5'000'000'000, [](RunnerConfig& c) {
        // A quarter of the learning rates by the end of the stage (docs/configuration.md).
        c.ppo_cfg.lr_schedule = LrSchedule::exponential(3'000'000'000, 5'000'000'000, 1.0f, 0.25f);
    }},
};
run_training_plan(config, plan, g_stop_training, "checkpoints");
```

A stage's `apply` changes the config the stage before trained with, so each stage only lists what it changes, and stage *i* trains with the base config after stages 0 to *i* applied theirs (`stage_config` returns it). Each stage builds a fresh runner, trains to its `until_steps` (the first update at or past it, like `train_until_steps`) and saves a checkpoint, and the next stage resumes from there, the same as stopping and restarting with the changed config: the model, optimizer state, PFSP pool and evaluation carry over, and a setting the evaluation depends on (such as `ticks_per_step`) restarts its curve as it would on a restart.

On startup the plan resumes in the stage the latest checkpoint is in, so after Ctrl+C or a crash, run the binary again and it carries on. Every stage's config is built first, and a stage that changes `bot_name`, `ac_cfg`, `max_players_per_team` or the mode fails at startup, since its resume would. Other mistakes in a stage's config (an invalid reward entry, say) only show when that stage starts, and stop the plan at the stage before's final checkpoint. A plan that's done trains nothing; append a stage to train further. Render, benchmark and calibrate modes run once with the config of the stage the latest checkpoint is in.

## Checkpoints & the PFSP Pool

```text
checkpoints/
├── <bot_name>_<steps>/
│   ├── model.pt        # Network weights and the settings the bot needs to run them
│   └── optimizer.pt    # Optimizer state
├── <bot_name>_pool/    # PFSP opponent pool
│   ├── <steps>.pt      # Snapshots, loadable and deployable like any model.pt
│   ├── state.txt       # Win records
│   └── meta.txt        # Network shape the pool was saved with
├── <bot_name>_eval/    # Evaluation anchors and results
│   ├── snapshot_<steps>.pt  # Anchors, loadable and deployable like any model.pt
│   ├── results.txt     # The learner's wins, losses, and draws against each anchor and Nexto rung, per evaluation and team size
│   ├── ratings.txt     # The learner's rating and its standard error, per evaluation and team size (× 173.7 for Elo)
│   └── meta.txt        # The settings the ratings depend on
└── nexto_ladder.txt    # Only with non-default kickoff jitter or game cap: the Nexto ladder measured for them (quick at startup, or precise with calibrate_nexto_ladder)
```

Checkpoints are written to a temporary directory and renamed when complete, so an interrupted save never produces a broken checkpoint. If the optimizer state can't be loaded (for example after switching optimizer presets), training continues with a fresh optimizer.

To roll back, delete the checkpoint directories newer than the one you want. On startup, pool snapshots, evaluation anchors, and evaluation results newer than the resumed checkpoint are deleted automatically. If the network shape changed, the old pool is moved aside to `<bot_name>_pool.stale-<time>` and a fresh pool is started. The evaluation directory is moved aside the same way, to `<bot_name>_eval.stale-<time>`, when a setting its ratings depend on changed (see [Evaluation](evaluation.md)), and the next evaluation places the learner on Nexto's ladder again (or, without Nexto, makes it the new reference). TensorBoard logs are appended, not rolled back, so move the old event file out of `logs/` if you want clean curves.
