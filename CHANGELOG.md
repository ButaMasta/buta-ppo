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