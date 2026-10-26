### Current Unreleased Changes
1. LayerNorm during autocast now correctly interprets features in bf16 since internally on the version of libtorch compiled with the project this is computed in fp32 for the accumulations so there is no noticeable precision loss, just faster compute times skipping the cast prior.
    - Non-breaking.
    - Training uses about 2.3× less VRAM, since activations saved for backward are no longer fp32 copies.
2. Truncated value calculation with rewards is now deferred to happen once per rollout versus the prior setup which was once per step.
    - Non-breaking. No changes here impact the user.
3. Added a rollout bottleneck diagnostic and optional CPU pinning.
    - The console prints `Phys Wait`, the share of the rollout the main thread spent waiting for the env workers, and TensorBoard logs it as `Performance/Rollout_Physics_Wait` (a fraction). A high value means physics is the limit, and a low one means the main thread is.
    - `RunnerConfig::main_thread_cpus` and `RunnerConfig::worker_cpus` pin the main thread and the env workers to lists of logical CPU ids. Both default to empty, which leaves the threads unpinned as before. A CPU this process can't run on fails at startup, and lists that share a CPU print a `[WARN]`.
    - Non-breaking. Existing configs run unpinned, as before.
4. Added a benchmark mode that finds the fastest thread count, CPU pinning, and env count for a config on the current machine.
    - `RunnerConfig::benchmark = true` measures instead of training, then prints a table and the recommended config values to paste in. Values the automatic settings already give are printed as `0`, and empty CPU lists mean unpinned. It asks for confirmation first, takes a few minutes, and loads, saves, and logs nothing. Each config runs real training updates, so the SPS shown are what training gets.
    - Each config runs under a light workload (a fresh policy) and a heavy one (scripted ball-chasing cars with extra ball predictions), since a bot's speed swings between the two as it learns to play the ball.
    - It recommends the configured env count unless one nearby scores at least 5 points higher, and the fastest unpinned CPU setup when it is within 2 points of a pinned one.
    - Non-breaking. It defaults to off, and training runs as before. The TensorBoard log file is now opened when training starts instead of when the runner is constructed.
5. Replaced `RunnerConfig::num_envs_per_thread` with `num_envs` and `target_steps_per_agent`, and made the thread count automatic.
    - `num_envs` sets the total env count, which no longer has to be a multiple of `num_threads`, since workers already take envs one at a time.
    - With `num_envs = 0` (the default), the env count is picked so each learner agent collects about `target_steps_per_agent` steps per update (default `100`).
    - `num_threads = 0` (the default) uses one worker per CPU in `worker_cpus`, or when that's empty, one per CPU the process may run on except one, so the main thread has a CPU of its own.
    - Breaking: configs that set `num_envs_per_thread` no longer compile. Set `num_envs` to the old `num_envs_per_thread * num_threads` to keep the same env count, or `target_steps_per_agent` to the steps per agent you want.
6. Added env cost telemetry to TensorBoard, aggregated over each log window like the other tags.
    - `Performance/Env_Step_Microseconds`: worker time per env step, including resets and copying out the results.
    - `Performance/Touch_Ticks_Per_Env_Step` and `Performance/Repredictions_Per_Env_Step`: ticks with a ball touch and full ball predictions per env step. A bot that keeps touching the ball costs about twice as much per step as a fresh one, mostly in ball prediction, so these explain most SPS drops during training.
    - Non-breaking.
7. Training warns when the CPU setup leaves speed unused, once per run each, judged from the median `Phys Wait` of the last 20 updates.
    - Physics is the limit (30% and up) while 2 or more CPUs have no env worker.
    - The main thread is the limit (10% and below) while it's unpinned and shares every CPU with the workers.
    - Non-breaking. Setups recommended by the benchmark trigger neither.
8. Warnings no longer scroll away under the update prints.
    - Every update ends with a ` | Warnings:` line listing a short label for each warning so far and when it was raised, and the full warnings are printed again when training ends.
    - Covers the reward config, CPU pinning, unusable optimizer state, and rollout bottleneck warnings.
    - Non-breaking.
9. The RLBot bot library now rebuilds when any C++ header or CMake file changes, not only when one of four `.cpp` files does.
    - Before, editing a header such as `default_obs.hpp` (which holds the obs coefficients) or `ffi.h` left the old `libbuta_ppo_bot.so` in place, so a deployed bot could build a different obs from the one it was trained on without any error.
    - `deploy.py` now copies the most recently built `libbuta_ppo_bot.so` and prints its path and build time. It used to copy whichever copy it found first, and Cargo keeps older copies from earlier builds next to the current one.
    - Non-breaking. Editing training-only C++ files now also reruns the bot build step, which adds a second or two to builds. Redeploy existing bots to make sure they use an up-to-date library.
10. Models are now checked against the configured network when they're loaded: resuming training, the PFSP pool, and the RLBot bot.
    - Every weight in the file must exist in the network with the same shape, and the other way around. A mismatch fails with an error naming the first weight that differs, e.g. `'shared_mlp.0.weight' has shape [1280, 226] in the file but [1280, 195] in the network`.
    - Before, `torch::load` took on the file's shapes without checking. A resume after changing `ac_cfg`, or a bot whose `bot_api.cpp` settings didn't match its checkpoint, either failed at the first forward pass (the bot then sent no inputs every tick) or silently ran the checkpoint's network instead of the configured one.
    - Weights are read onto the network's device, so resuming from a checkpoint saved on another device (e.g. the CPU) works too. The bot already did this.
    - Non-breaking for matching checkpoints. Resolution for a mismatch: on resume, change `bot_name` or move the old checkpoints to start fresh. In the bot, set `bot_api.cpp` to the config the checkpoint was trained with. A PFSP snapshot that doesn't match is skipped with a message.
11. Hardening and small fixes.
    - `SIGTERM` now stops training like Ctrl+C: the current update finishes and a checkpoint is saved. Before, process managers (`systemctl stop`, `docker stop`, Slurm) killed the run and lost everything since the last checkpoint.
    - A `[WARN]` when `num_minibatches` doesn't divide the steps per update evenly and the rollout grows by more than 5% to make it fit. It names the nearest minibatch count that needs no growth. Some counts used to grow the steps, time and memory per update several times over without a message.
    - Errors instead of silent corruption for invariants that hold today but were only implied: the training graph checks that every minibatch has the size it was captured with, and `DefaultAction` checks that its loops build exactly `ACTION_SPACE_SIZE` actions.
    - The tensorboard_logger patch fails the configure if `add_scalar` doesn't come out with a 64-bit step, so a changed upstream signature can't silently bring back negative steps past 2^31.
    - `rs_arena_set_car_controls` ignores an out of range car index, like `rs_arena_set_car_state`, instead of panicking and aborting the process.
    - Corrected the path comments at the top of 10 source files.
    - Non-breaking.
12. The no-touch timeout and the episode time limit are now config values, `RunnerConfig::no_touch_timeout_seconds` (default `10`) and `RunnerConfig::episode_time_limit_seconds` (default `150`). Both used to be fixed.
    - `0` turns a limit off. A negative or non-finite value stops training at startup, and a no-touch timeout longer than the time limit prints a `[WARN]`, since it can never end an episode.
    - Non-breaking. The defaults match the old fixed values, so existing configs behave the same.
13. `util::math::euler_to_mat3` now uses the game's angle convention, the same as RLBot's `Rotator`, RLGym, and RocketSim: positive pitch raises the nose, positive yaw turns toward the car's right, and positive roll lowers its right side. Added `util::math::mat3_to_euler`, its inverse, for working in angles in setters, rewards, and metrics.
    - Before, it used the textbook right-handed convention, which has the opposite sign for pitch and roll, so a setter asking for the nose 30° up got it 30° down.
    - Breaking for custom code that calls `euler_to_mat3` with a nonzero pitch or roll. Resolution: to keep the old orientation, negate both, `euler_to_mat3(-pitch, yaw, -roll, ...)`. Code that passes angles from RLBot or RLGym needs no change, and is now correct.
    - The built-in random state setter is unaffected: its pitch and roll ranges are symmetric, so it spawns the same distribution of orientations as before. Existing checkpoints are unaffected.
14. Each rollout step's results are now written into the rollout buffer by one replay of a CUDA graph per env group, in place of 11 separate GPU operations launched from the main thread.
    - The demo config trains about 1% faster on an i5-14600K with an RTX 5070. Setups where the main thread is the limit (a low `Phys Wait`) gain the most, and setups where physics is the limit gain nothing.
    - The rollout buffer, advantages, and training come out bit-identical to before. The graph tracks the buffer step on the GPU, and that count is checked against the CPU's at the end of every rollout, so a step written twice or not at all stops training with an error instead of leaving stale data in the buffer.
    - Non-breaking.
15. The per-tick arena state read is cheaper: clearing the state before each read now costs a fraction of what it did, with every field read exactly as before.
    - Setups where physics is the limit (a `Phys Wait` around 50% or more) train about 0.7% faster.
    - Non-breaking.
16. Saved models now store the settings they were trained with (the `ac_cfg` network config, `max_players_per_team`, and `ticks_per_step`), and the RLBot bot sets itself up from them, so its settings no longer have to be matched to the training config by hand.
    - `src/bot_api.cpp` and `rust_bridge/src/main.rs` no longer hardcode the layer sizes, layer norm, max players per team, or tick skip. One build of the bot runs any model, and prints the settings it read on startup. A bot built from obs or action code whose obs or action size differs from the model's stops at startup with both sizes.
    - `python deploy.py` now also copies the model and the collision meshes into `bot/`. By default it deploys the latest checkpoint; pass a checkpoint directory or a `.pt` file (such as a PFSP pool snapshot) to deploy another, and `--bot <bot_name>` when `checkpoints/` holds more than one bot.
    - The bot runs the model named by the first argument of its run command, or `model.pt` without one, so copies of `bot.toml` can put several versions of a bot in RLBot.
    - Resuming training from a checkpoint with a different network config or `max_players_per_team` now fails with the checkpoint's settings next to the configured ones, instead of naming the first mismatched weight. A changed `ticks_per_step`, which used to pass silently, prints a `[WARN]`.
    - Breaking for deploying models saved before this change: they have no stored config, so the bot and `deploy.py` refuse them. Training still resumes from them, and checkpoints and pool snapshots saved from then on store it. Resolution: to re-save one for deploying, make its checkpoint the latest in `checkpoints/` and run training once with the update count in `runner.run(...)` set to `0`, which saves it again in place with the same weights and optimizer state. For a checkpoint that isn't the latest, move the newer ones out first and set `pfsp_cfg.envs = 0` for that run, since resuming deletes pool snapshots newer than the resumed checkpoint.
    - Breaking for custom code calling the bot's C API: `bot_api_max_players_per_team` now takes the context returned by `bot_api_init`, and `bot_api_tick_skip(ctx)` was added.
17. Editing C++ files no longer rebuilds the Rust bridge. An edit to `main.cpp` builds in about 5 s instead of about 22 s.
    - The RLBot bot binary is now its own package, `rust_bridge/bot/`, in a Cargo workspace with the bridge. The training build only builds the bridge, which no longer has a build script, so it no longer builds the bot binary and its C++ library on every C++ edit, and no longer needs LibTorch.
    - `deploy.py` builds the bot with `cargo build --release -p buta_ppo_bot`, and still rebuilds it when any C++ file changes.
    - Non-breaking, unless you build the bot by hand: use `cargo build --release -p buta_ppo_bot` in `rust_bridge/` in place of `--bin buta_ppo_bot`. Its sources moved to `rust_bridge/bot/src/`.
18. The RLBot bot now runs inference on the CPU on one thread, and `bot/` carries every LibTorch library it needs, so it can be moved or copied to another x86-64 Linux machine without LibTorch, CUDA, or a GPU.
    - Before, the bot loaded every LibTorch and CUDA library from the LibTorch path it was built against, and the four libraries `deploy.py` copied into `bot/libs/` (429 MB) were never loaded. It only ran on a machine with LibTorch at that path.
    - `deploy.py` now empties `bot/libs/` and copies the libraries the bot's C++ library actually loads: `libtorch_cpu`, `libc10`, and their dependencies, about 440 MB.
    - With the demo network, a decision took about 0.6 ms, and at most 2.5 ms, on an i5-14600K with every core kept busy, against 0.5 ms (at most 5.5 ms) for the old GPU bot. LibTorch's default multi-threaded CPU inference went past the 8.33 ms between game ticks on every decision under the same load, so the bot uses a single thread. The first decision no longer takes about 130 ms while CUDA starts up.
    - The bot's C++ library is no longer built with `-march=native`, so it runs on other CPUs. The machine running the bot needs a glibc at least as new as the build machine's.
    - Non-breaking. Rerun `deploy.py` to repackage an existing bot.
19. Added external bots: bots from outside the framework that drive a whole team in the visualizer, starting with Nexto, the RLGym community bot, included with permission from its author (see `THIRD_PARTY_NOTICES.md`).
    - `RunnerConfig::render_external_bots` takes `{bot, team}` pairs, e.g. `{{create_nexto_bot("src/external/nexto/nexto-model.pt"), ffi::Team::Orange}}` from `src/external/bots.hpp`. Teams not listed play the policy. Render only for now: listing one outside render mode stops at startup with an error.
    - Nexto plays as the RLBot botpack version does: its own observations and action table, its decision timing (a decision every 8 ticks, taking effect 6 ticks after the state it was made from), and its scripted speedflip kickoff. It plays through the 3 second frozen countdown before each kickoff as it would in game, each car keeps its own decision timing, and bots keep their state across goals like a bot staying in a match. A tick-by-tick replay of its packets through the original Python bot gives the same controls, within 1e-8. `NextoOptions` sets how random it plays (`beta`) and its kickoff behavior.
    - Its network is rebuilt in C++ from the original model's weights and runs on the GPU by default (`NextoOptions::device`), at about 0.24 ms per decision. TF32 is off for Nexto's own matmuls, since it changed 0.4% of its choices.
    - `ticks_per_step` doesn't have to match Nexto's 8. Other bots can be added through `src/external/external_bot.hpp`.
    - Non-breaking.
20. The visualizer now plays the policy's most likely action, as the RLBot bot does, so it shows how the bot plays in a match. `RunnerConfig::render_best_action = false` samples actions like training, as before.
    - Non-breaking, but the visualizer plays differently from before by default.
21. Added evaluation, which tracks the bot's skill over training. Every `RunnerConfig::eval_cfg.steps_per_eval` steps (default 50M), training pauses for a few seconds while the learner plays fixed past versions of itself, the anchors, and TensorBoard gets a Bradley–Terry Elo rating per team size under `Skill/`.
    - The games: a fixed bank of kickoffs, each car moved by up to 2.5 uu and turned by up to 0.01 rad, each played once from each side against each anchor, from the kickoff to the first goal or `game_cap_seconds` (a draw). Both sides play their most likely action. The same models always give the same results.
    - The first evaluation makes the learner the reference anchor at 0 Elo. A later learner becomes an anchor once it scores `promote_win_rate` (0.7) against the newest one, over all team sizes. Anchor ratings never change, so earlier points on the curve never move. An anchor the learner is predicted to score above `retire_win_rate` (0.85) against isn't played until the learner falls back, and the newest one is always played.
    - Each team size has its own rating. `eval_cfg.team_sizes` defaults to every even layout in `match_distributions`.
    - Anchors, every result, and every rating are kept in `checkpoints/<bot_name>_eval/`, rolled back with the checkpoints like the PFSP pool. Changing a setting the ratings depend on moves the directory aside to `<bot_name>_eval.stale-<time>`, and evaluation starts over with a new reference.
    - An evaluation took about 3 s with two anchors played and three team sizes on an i5-14600K with an RTX 5070, about 3% of training time at the default interval. `Performance/Eval_Time` logs it, and it isn't counted in the steps per second. The evaluation games run on their own worker threads, on `worker_cpus` like training.
    - Non-breaking, but on by default: a run without an evaluation directory makes its current learner the reference at the start of training. `eval_cfg.steps_per_eval = 0` turns evaluation off.
22. Small simulation changes that came with evaluation.
    - The ball prediction in the observations now clears the prediction sim's cached contacts on every reprediction, as the ball_sim crate recommends when one sim is reused for predictions, so a prediction no longer depends on where the previous one ended. On 57,600 recorded steps it gave the same predictions as before, bit for bit, so existing bots are unaffected.
    - `ffi::seed_arena_rng` seeds the RocketSim arena's random number generator, which picks where a demolished car respawns. Evaluation uses it so its games repeat exactly. Training still seeds it randomly.
    - The env worker threads now run in `util::WorkerPool` (`src/util/worker_pool.hpp`), shared by training and evaluation. Training speed is unchanged.
    - Non-breaking.
23. Added a `VelocityBallToGoal` team reward and updated the demo config in `main.cpp`.
    - `create_velocity_ball_to_goal_reward()` (`src/reward/velocity_ball_to_goal.cpp`) gives each team the ball's velocity toward the back of the opponent's net, divided by the ball's max speed of 6000, so from −1 to 1 per tick. Every player on the team receives it.
    - The demo config now trains 1v1, 2v2, and 3v3 (45%, 30%, and 25% of the envs) with `max_players_per_team = 3`, adds `VelocityBallToGoal` with a weight of 0.5, and raises `Goal` from 10 to 50. These are still demo values, not tuned.
    - Non-breaking: only the demo values changed, so configs of your own are unaffected. A run started from the old demo config can't resume with the new one, since `max_players_per_team` changes the observation size.
24. Evaluation now rates on a fixed scale: Nexto, the RLGym community bot, playing its best action is 1500 Elo in 1v1, 2v2, and 3v3. `eval_cfg.nexto`, on by default.
    - Each evaluation also plays, per team size, one rung of a ladder of 13 weaker versions of Nexto (more random play, `beta` from 1 down to 0): the rung closest to the learner's rating. The rungs' ratings were measured from Nexto-vs-Nexto games and ship in `src/external/nexto/nexto_ladder.txt`. With other kickoff jitter or `game_cap_seconds` than the defaults, the ladder is measured once at startup (about 90 s, announced) and saved as `checkpoints/nexto_ladder.txt`. `RunnerConfig::calibrate_nexto_ladder = true` regenerates the shipped file (about 20 min).
    - The first evaluation of a run places the learner on the ladder with a binary search, up to 4 rounds of Nexto games, and makes it the first anchor at its fitted rating instead of 0. Team sizes above 3, which Nexto doesn't play, keep the first anchor as their reference at 0. Promotion and retirement of anchors work as before.
    - Nexto plays as in the visualizer, with its own decision timing, scripted kickoff, the countdown, and each car joining at its own tick, but with fresh controllers in every game, so its games repeat exactly like the anchors'. Its inference is batched over every game and up to 6 ticks at once (Nexto first uses an action 6 ticks after deciding it) through CUDA graphs, giving the same games, bit for bit, as each game running Nexto on its own.
    - Cost, on an i5-14600K with an RTX 5070 and three team sizes: an evaluation with one anchor took about 4.5 s, against about 2 s without Nexto, and the first one about 11 s. Nexto's games take longer against weaker rungs, whose games last longer. Nexto adds about 550 MiB of GPU memory, 1.2 GB of RAM, and 1 s at startup.
    - Breaking for existing evaluation directories: the old ratings aren't on Nexto's scale, so on the first start `<bot_name>_eval/` is moved aside to `<bot_name>_eval.stale-<time>` and the Elo curve starts over. Resolution: none needed. `eval_cfg.nexto = false` keeps the first-anchor reference, and the existing directory, exactly as before.
    - For custom external bots: `ExternalBot` gained `model()` and `make_batch_runner()`, and `ExternalController` gained `decision_delay()`, `request_decision()`, and `decision_interval()`, all with defaults that keep the old behavior. `NextoBot::with_options` makes Nexto with other options on the same network.
    - `Skill/<N>vN_Score`, `Draw_Rate` and `Mean_Game_Seconds` count only the games against snapshot anchors, as before Nexto, and are skipped on a run's first evaluation, which plays only Nexto. New tags `Skill/<N>vN_Nexto_Beta`, `Nexto_Score`, `Nexto_Gap` and `Nexto_Gap_Std_Error` show the rung played, the score against it, and whether the Nexto games and the snapshot games agree on the learner's rating (logged while the score against the rung is between 0.05 and 0.95).
    - `RocketSimEnv::reset` and `step` were split into parts shared with evaluation's play. Training and the visualizer give bit-identical results, checked against a trace from before the change, at the same speed.

### Releases
#### 0.2.0
1. Added metrics logging via a custom system. The user can implement an abstract class to create their own metrics.
    - Non-breaking. No resolutions need to be made.
2. Added `steps_per_log` to `RunnerConfig` to set how many lifetime steps pass between TensorBoard logs. Each log aggregates every update since the previous one, and a partial window is logged when training stops. The default of `0` logs every update, as before. Console output is unchanged.
    - Non-breaking. Existing configs keep logging every update.
    - The per-slot `PFSP/Slot_<n>_Version` and `PFSP/Slot_<n>_Win_Rate` tags are no longer logged, since a slot plays a different snapshot each update. `PFSP/Win_Rate`, `PFSP/Games`, and `PFSP/Pool_Size` remain, and the console still lists each update's opponents.
    - `Performance/Rollout_SPS` no longer includes time spent writing logs, so it reads slightly higher than before.
3. Added an optional `pre_tick(agents, current_state, previous_state)` method to `RewardFunction`. It is called once per physics tick before `get_reward` is called for any agent, so rewards can compute state shared across agents for that tick.
    - Non-breaking. The default implementation does nothing, so existing rewards need no changes.
4. Exposed RocketSim's car-to-car bump events. `ArenaState::events.car_bumped_car[bumper][victim]` has `bumped` set when a car bumps another car that tick, and `is_demo` set when that bump demolished the victim. Also added an "Arena State Reference" section to the README describing the fields available in `ArenaState`.
    - Non-breaking. The new field is added at the end of `rs_arena_events`, and both sides of the FFI were updated together.
5. Added zero-sum, team-sharing, and team rewards.
    - `RewardEntry` takes an optional `RewardOptions`, applied to each step's weighted values:
        - `self_share` (default `1`) is the fraction of an agent's own reward it keeps. The rest is split among its teammates, so the agent keeps the same fraction in every match layout.
        - `opp_scale` (default `0`) subtracts the opponent team's reward. At `1`, the reward is zero-sum between the teams, including uneven ones.
        - Example: `{"TouchBall", create_touch_ball_reward(), 1.0f, {.self_share = 0.7f, .opp_scale = 1.0f}}`.
    - Rewards that only depend on the team can subclass `TeamRewardFunction` and implement `get_team_reward(team, ...)`, which is called once per team per tick. `GoalReward` is now a team reward that returns 1 to the scoring team, and the demo config uses it with `opp_scale = 1`.
    - Reward entries are checked at startup. Invalid options, non-finite weights, and duplicate reward names throw. Settings that have no effect (such as `self_share` on a team reward or in 1v1-only layouts, or a weight of 0) and an `opp_scale` above 1 print a `[WARN]`.
    - New TensorBoard groups:
        - `Reward_Magnitude/<name>`: the average absolute value per step that agents received, after the options.
        - `Reward_Share/<name>`: each reward's fraction of the total magnitude.
        - `Reward/Mean_Abs_Step`: the total magnitude.
    - `Reward_Components/` and `Reward/Mean_Step` now log values before the options are applied.
    - Non-breaking. Existing rewards and configs work unchanged, except for configs with duplicate reward names, which now throw since their telemetry used to merge silently. Resolution: give each entry a unique name.
    - `GoalReward` no longer returns −1 on its own, so it only punishes the conceding team when paired with `opp_scale = 1`. Configs that use it without `opp_scale` lose the conceding penalty. Resolution: add `{.opp_scale = 1.0f}` to the entry.
    - `Reward_Components/Goal` now logs goals scored, where it used to read about 0 during self-play, so it rises when resuming a run logged before this change.

#### 0.1.1
1. Updated the logging with a patch via cmake to replace the dependency's invalid `int32_t` logging for the x-axis to be `int64_t`. 
    - Non-breaking change but existing `.tfevents` files with already faulty negative values cannot be retroactively updated.
    - Resolutions:
        - Deleting the file is an accepatable resolution.
        - Keeping it will just mean the new values get added as expected.

#### 0.1.0
Initial release.