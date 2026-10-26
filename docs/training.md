# Training

On startup, the framework searches `checkpoints/` for the most recent checkpoint matching the configured `bot_name`. If one is found, training resumes from its step count. Otherwise, a fresh model is initialized. A checkpoint has to match the configured network: if the `ac_cfg` layers or layer norm changed, or the obs size did (through `max_players_per_team` or the obs builder), startup fails with an error showing the settings the checkpoint was trained with next to the configured ones. Change `bot_name` or move the old checkpoints out of `checkpoints/` to start fresh. A changed `ticks_per_step` only prints a `[WARN]`, and training continues with the new value.

A run trains for at most the number of updates passed to `runner.run` in `main.cpp` (`1'000` by default, about 100M steps with the demo config), then saves a checkpoint and exits. Run the binary again to continue from there.

To stop training safely, press Ctrl+C, or send `SIGTERM` (what `kill`, `systemctl stop`, and `docker stop` send). The current update finishes, a checkpoint is saved, and the program exits.

Checkpoints are also saved every `config.steps_per_checkpoint` lifetime steps (`100'000'000` by default; `0` turns this off), so a crash only loses the progress since the last one. Every checkpoint is kept, and each one is roughly as large as the network and its optimizer state (about 50 MB for the demo network), so delete old ones you don't need.

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
