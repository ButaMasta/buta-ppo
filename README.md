# buta-ppo

buta-ppo is a C++ reinforcement learning framework implementing Proximal Policy Optimization (PPO) for training Rocket League bots. Built using LibTorch and [RocketSim](https://crates.io/crates/rocketsim).

My main goal with this framework was to make use seamless and speeds respectable. Comparing against my own experience with [GigaLearn](https://github.com/ZealanL/GigaLearnCPP-Leak) I can confidently say it's at least four times as fast. As for training stability and quality though I cannot say as this project is so new I have not extensively tested it though there are some bots which have surpassed nexto level utilizing this framework already.

For reference, the demo configuration in `src/main.cpp` (1v1, 500 envs, a ~6M parameter network, PFSP enabled, CPU pinning) trains at about 530k steps per second with a fresh policy, and about 480k once the bot begins to properly interact with the ball, on an i5-14600K with an RTX 5070. Steps here are agent steps: each learner car's step counts once, so a 1v1 env step is two steps, or one when a PFSP opponent drives the other car. The console's `Total SPS` and TensorBoard's `Performance/Total_SPS` count the same way.

## Features

* **PPO:** GAE advantages, clipped surrogate objective, entropy bonus, advantage normalization over the whole rollout, and global gradient clipping.
* **Pipelined Rollout:** Environments are split into two groups. While worker threads step one group's physics, the main thread runs inference for the other, so the CPU and GPU stay busy at the same time.
* **CUDA Graphs:** Rollout inference, each rollout step's writes into the buffer, and the entire training minibatch step (forward, loss, backward, gradient clipping, optimizer step) are captured once and replayed, removing kernel launch overhead. Training runs under bf16 autocast, and rollout inference uses TF32 matmuls.
* **Zero Mid-Step Allocations:** Observations, action masks, rewards, and actions live in persistent, pinned buffers allocated before training starts.
* **Graph-Safe Optimizers:** Adam, AdamW, and Muon (on hidden weight matrices), reimplemented so they can run inside a CUDA graph. They can be configured per network section and parameter role.
* **PFSP:** Prioritized fictitious self-play pits a fraction of the environments against past versions of the learner, chosen by the learner's win rate against them.
* **Observations & Actions:** Team-relative observations with X-axis mirroring, ball prediction, and shuffled car slots. A 90-action discrete lookup table with action masking based on the car's state.
* **Rust FFI Bridge:** Physics runs in RocketSim through a lightweight C ABI exported from Rust.
* **Telemetry & Checkpoints:** TensorBoard logging with custom per-tick metrics, automatic resume from the latest checkpoint, and a persistent PFSP opponent pool.
* **RLBot Deployment:** A Rust RLBot v5 bot binary that runs trained checkpoints in real matches.

## How It Works

Each update runs through the following stages:

1. **Rollout:** `VecEnv` steps every `RocketSimEnv` on a thread pool. Each env applies its agents' actions, ticks the arena `ticks_per_step` times while accumulating rewards and metrics, and builds the next observations and action masks. Finished episodes are reset automatically.
2. **Inference:** Each env group's policy forward pass and action sampling run as a single CUDA graph replay that reads directly from `VecEnv`'s batched buffers.
3. **Advantages:** `RolloutBuffer` stores every step on the GPU, then computes GAE advantages and returns. Truncated episodes are bootstrapped with the critic's estimate of the final state.
4. **Training:** `PPOTrainer` shuffles the rollout into minibatches and replays the captured training graph for each one.
5. **Checkpointing:** When training ends or is interrupted, the model and optimizer state are saved, and they are loaded automatically on the next start.

## Directory Structure

```text
buta-ppo/
├── assets/
│   └── collision_meshes/    # Required RocketSim arena geometry (not tracked)
├── bot/                     # RLBot bot directory, filled by deploy.py (only the .toml templates are tracked)
├── checkpoints/             # Saved models, optimizer state, and the PFSP pool
├── include/buta_ppo/        # FFI declarations shared with the Rust bridge
├── logs/                    # TensorBoard event files
├── rust_bridge/             # Rust FFI wrapper for RocketSim and the RLBot bot binary
├── src/
│   ├── action/              # Discrete action table and action masks
│   ├── env/                 # Single environment and the vectorized env pool
│   ├── external/            # Bots from outside the framework (Nexto) that can drive a team in the visualizer
│   ├── metric/              # Metric functions and the metric manager
│   ├── obs/                 # Observation builder and ball prediction
│   ├── reward/              # Reward functions and the reward manager
│   ├── rl/                  # Network, rollout buffer, trainer, optimizers, PFSP, and the training runner
│   ├── state/               # State setters for episode resets
│   ├── util/                # Inline math and vector operations
│   ├── bot_api.cpp          # Inference-only library used by the RLBot bot
│   └── main.cpp             # Entry point and all training configuration
├── deploy.py                # Builds and packages the RLBot bot
└── THIRD_PARTY_NOTICES.md   # Notices for included third-party material (Nexto)
```

## Prerequisites

* **OS & Compiler:** Linux with a C++20 compiler (GCC or Clang).
* **GPU:** An NVIDIA GPU with CUDA. CPU training is not supported. Training runs in bf16, so an Ampere (RTX 30-series) or newer GPU is recommended. Older GPUs are untested.
* **LibTorch:** The CUDA-enabled C++ distribution, located through the `LIBTORCH` environment variable (see [Build Instructions](#build-instructions)). The trainer relies on recent LibTorch internals (fused optimizer kernels, CUDA graphs), so older versions may not build.
* **CUDA Toolkit:** A version compatible with your LibTorch build, needed when CMake configures LibTorch.
* **CMake:** Version 3.20 or higher. Ninja is recommended.
* **Rust Toolchain:** Cargo with Rust 2024 edition support (1.85 or newer).
* **Protobuf & Abseil:** Development packages, needed by the TensorBoard logger.
* **Collision Meshes:** Rocket League arena geometry placed in `assets/collision_meshes/`. See RocketSim's documentation for how to obtain them.
* **Python 3:** For TensorBoard and `deploy.py`.

Tested with:

| Component | Version |
|---|---|
| GPU | RTX 5070 |
| LibTorch | 2.14.0 (CUDA 13.2 build) |
| CUDA Toolkit | 13.4 |
| GCC | 16.2 |
| CMake | 4.4 |
| Rust | 1.99 |

## Build Instructions

CMake fetches the remaining C++ dependencies and builds the Rust FFI bridge as part of the build.

0. **Clone the Repository:** The `stable` branch always points to the latest release, while `main` has in-progress development.
```bash
git clone -b stable https://github.com/ButaMasta/buta-ppo.git
cd buta-ppo
```
To use a specific release instead, clone its tag, e.g. `git clone -b v0.1.0 https://github.com/ButaMasta/buta-ppo.git`.

1. **Set the LibTorch Path:** Set the `LIBTORCH` environment variable to your LibTorch directory, the one containing `lib/` and `share/`. Add it to your shell's config so it persists:
```bash
export LIBTORCH=/path/to/libtorch         # bash / zsh, e.g. in ~/.bashrc
set -Ux LIBTORCH /path/to/libtorch        # fish, saved automatically
```
CMake reads it when it configures and remembers the path, so later builds work without it. `deploy.py` needs it set whenever you run it. To switch to a different LibTorch, set `LIBTORCH` to the new path and run the configure command in step 3 again.

2. **Add the Collision Meshes:** Place your `collision_meshes/` directory inside `assets/`. The `logs/` and `checkpoints/` directories are created on the first run.
```bash
mkdir -p assets
```
3. **Configure and Build:**
```bash
cmake -G Ninja -B build
ninja -C build
```
4. **Subsequent Builds:** From the project root you only need to run:
```bash
ninja -C build
```

For a debug build, swap the commented `-g -O0` `target_compile_options` blocks in `CMakeLists.txt`.

### Updating

Read the release notes first, since a minor release may break existing checkpoints (see [Versioning](#versioning)). Your configuration lives in `src/main.cpp`, so commit your changes before pulling and Git will merge them with the update:
```bash
git commit -am "My config"    # skip if you haven't changed anything
git pull
ninja -C build
```
`git describe --tags` shows which release you're on.

## Running Training

Run the compiled binary from the project root, so the relative paths to `assets/`, `checkpoints/`, and `logs/` resolve:
```bash
./build/buta_ppo
```

On startup, the framework searches `checkpoints/` for the most recent checkpoint matching the configured `bot_name`. If one is found, training resumes from its step count. Otherwise, a fresh model is initialized. A checkpoint has to match the configured network: if the `ac_cfg` layers or layer norm changed, or the obs size did (through `max_players_per_team` or the obs builder), startup fails with an error showing the settings the checkpoint was trained with next to the configured ones. Change `bot_name` or move the old checkpoints out of `checkpoints/` to start fresh. A changed `ticks_per_step` only prints a `[WARN]`, and training continues with the new value.

A run trains for at most the number of updates passed to `runner.run` in `main.cpp` (`1'000` by default, about 50M steps with the demo config), then saves a checkpoint and exits. Run the binary again to continue from there.

To stop training safely, press Ctrl+C, or send `SIGTERM` (what `kill`, `systemctl stop`, and `docker stop` send). The current update finishes, a checkpoint is saved, and the program exits.

Checkpoints are also saved every `config.steps_per_checkpoint` lifetime steps (`100'000'000` by default; `0` turns this off), so a crash only loses the progress since the last one. Every checkpoint is kept, and each one is roughly as large as the network and its optimizer state (about 50 MB for the demo network), so delete old ones you don't need.

## Configuration

All configuration lives in `src/main.cpp`, in the `RunnerConfig` struct. The values there are demo values that show the available options, not tuned hyperparameters.

### Matches, State Setters, Rewards & Metrics

```cpp
// Match types, each with a weight for its share of the envs.
config.match_distributions = {
    {1, 1, 0.5f},   // 1v1 at 50% weight.
    {2, 2, 0.5f}    // 2v2 at 50% weight.
};
// The obs has a fixed number of car slots, so this must fit the largest team.
config.max_players_per_team = 2;

// How episodes start, each with a weight.
config.setter_distributions = {
    {create_default_kickoff_setter(), 0.30f},               // Regular kickoffs at 30%.
    {create_random_state_setter(true, true, true), 0.35f},  // Random, with cars on the ground, at 35%.
    {create_random_state_setter(true, true, false), 0.35f}  // Fully random at 35%.
};

// Besides a goal, episodes end after this long without a touch, or once they last this long. 0 turns a limit off.
config.no_touch_timeout_seconds = 10.0f;
config.episode_time_limit_seconds = 150.0f;

// Rewards with their names (for logging) and weights. Rewards are computed every physics tick.
config.reward_entries = {
    {"VelocityToBall", create_velocity_to_ball_reward(), 0.1f},
    {"TouchBall", create_touch_ball_reward(), 1.0f},
    {"Goal", create_goal_reward(), 10.0f, {.opp_scale = 1.0f}}  // Zero-sum, see Reward Options below.
};

// Optional metrics with their names (for logging). Metrics are computed every physics tick but don't affect training.
config.metric_entries = {
    {"InAirRatio", create_in_air_ratio_metric()}
};
```

**Episode Limits:** Besides a goal, an episode ends when no car has touched the ball for `no_touch_timeout_seconds`, or once it has lasted `episode_time_limit_seconds`. These are truncations, not terminations: the critic's estimate of the last state stands in for the rest of the episode, so an episode that is cut short isn't treated as worth nothing from there on. A short no-touch timeout keeps slow, ball-less play out of the training data, but the bot then rarely learns from it. `Episode/Truncations_No_Touch` and `Episode/Truncations_Time_Limit` in [Telemetry](#telemetry) show how often each limit ends episodes. `0` turns a limit off, a negative value stops training at startup, and a no-touch timeout longer than the time limit prints a `[WARN]`, since it can never fire.

Rewards are summed over the `ticks_per_step` ticks of each step, so per-tick rewards scale with `ticks_per_step`.

**Reward Options:** Each entry can take an optional `RewardOptions` that mixes its values between agents. They are applied to each step's weighted values, after the reward is computed for every agent.

```cpp
{"TouchBall", create_touch_ball_reward(), 1.0f, {.self_share = 0.7f, .opp_scale = 1.0f}}
```

- `self_share` (default `1`, in `[0, 1]`): the fraction of an agent's own reward that it keeps. The rest is split evenly among its teammates. An agent on team *T* with *n* players receives `self_share · r + (1 − self_share) · (mean of its teammates' r)`. The team's total is unchanged, and the agent keeps the same fraction of its own reward in every match layout, but each teammate's share shrinks as teams get bigger. Agents without teammates keep their whole reward.
- `opp_scale` (default `0`, `≥ 0`): subtracts the opponent team's total, split across the agent's team: `r − opp_scale · S_opp / n`. At `1`, the two teams' totals are exactly opposite (zero-sum), including with uneven teams. A team with no opponents (for example 1v0) is unaffected.

| `self_share = 0.5` | Agent keeps | Each teammate gets |
|---|---|---|
| 1v1 | all of it | — |
| 2v2 | half | half |
| 3v3 | half | a quarter |

The order of the two options doesn't matter. Don't set `opp_scale` on a reward that already punishes the other team (returning −1 to the team that conceded, for example), since that doubles it. `Reward_Magnitude/` in [Telemetry](#telemetry) shows when a reward is larger than expected.

Invalid options, non-finite weights, and duplicate reward names stop training at startup. Settings that have no effect, such as `self_share` in 1v1-only layouts, a zero weight, or an `opp_scale` above 1, print a `[WARN]` instead.

Metrics are averaged over every tick since the previous TensorBoard log, so they don't depend on `ticks_per_step`. Leaving `metric_entries` empty disables them. See [Extending the Framework](#extending-the-framework) to write your own and [Telemetry](#telemetry) for how they are logged.

### Environments & Batch Size

```cpp
config.num_threads = 0;             // Worker threads stepping the envs. 0 = one per CPU, minus one.
config.num_envs = 0;                // Total envs. 0 = picked from target_steps_per_agent. Training needs at least 2.
config.ticks_per_step = 8;          // Physics ticks (120 per second) each action is held for.

config.target_steps_per_update = 50'000;
config.num_minibatches = 1;
config.target_steps_per_agent = 100; // Steps each agent collects per update, which sets the env count.
```

With `num_envs = 0`, the env count is the one whose steps per agent (`target_steps_per_update` divided by the learner agents) come closest to `target_steps_per_agent`. More envs are usually faster, but each update then learns from shorter stretches of play per agent. Setting `num_envs` above 0 uses that count instead. Workers take envs one at a time as they finish, so `num_envs` doesn't need to be a multiple of `num_threads`, and `num_threads = 0` uses one worker per CPU in `worker_cpus`. When that's empty, it uses one per CPU the process may run on except one, which leaves the main thread a CPU of its own.

The actual steps per update are rounded up from `target_steps_per_update`, so every agent contributes the same number of steps and the total divides evenly into `num_minibatches`. The update size is limited by VRAM. Start with a small `target_steps_per_update` or more minibatches, and increase from there.

#### Benchmarking Your Hardware

The fastest thread count, CPU pinning, and env count depend on the CPU, the network size, the match layouts, and how the bot plays. A fresh bot spends most of its time in the air, which is cheap to simulate, so the main thread's inference is the limit. A bot that keeps touching the ball costs about twice as much per step, mostly in ball prediction, so physics becomes the limit. A config tuned on a fresh bot can lose 15% or more once the bot learns to play the ball.

Benchmark mode measures this for you instead of training:

```cpp
config.benchmark = true;
```

It asks for confirmation, then takes a few minutes (about 3 for the demo config). Nothing is loaded, saved, or logged. Each config is measured under two workloads:

- **Light:** a fresh policy with this config's network, mostly airborne.
- **Heavy:** every car is a scripted ball-chaser, with extra ball predictions to match a bot that keeps contact with the ball. It is the same on every machine, so it doesn't need a trained checkpoint.

The benchmark first compares CPU setups found from the CPU's layout: the current config, a few unpinned thread counts, and the main thread pinned to the fastest core. Then it tries a few env counts around the configured one (`num_envs`, or the one picked from `target_steps_per_agent`) for that setup. Each config runs real training updates, rollout and training both, since rollouts back to back without the training in between run at a different speed on some setups. Each row shows the total SPS under both workloads, the steps per agent, and `Worst`, the lower of the two SPS as a share of the best in its column. The recommended CPU setup is the one with the best `Worst`, or the best unpinned one when it is within 2 points, since pinned CPU ids don't carry over to other machines. The recommended env count is the configured one, unless an env count with steps per agent within about a third of it scores at least 5 points higher; the output says when one scored higher but not by enough. It is all printed ready to paste:

```
Recommended config:
    config.num_threads = 0;             // 19 workers on this machine.
    config.num_envs = 0;                // 500 envs from target_steps_per_agent = 100.
    config.main_thread_cpus = {};       // Unpinned.
    config.worker_cpus = {};            // Unpinned.
```

Values that the automatic settings already give are printed as `0`, so they keep fitting if the machine or `target_steps_per_update` changes, and empty CPU lists mean unpinned. Replace all four lines in your config, including clearing CPU lists from an earlier pinned setup.

Run the benchmark the way you train, with whatever usually runs alongside (a browser, a video, Discord) open, since those take CPU time from the workers.

Steps per agent are a learning choice, so the benchmark only moves away from yours for a clear gain, and the table shows what other env counts would gain. The training time and peak VRAM are printed too. They depend on `target_steps_per_update`, `num_minibatches`, and the network, not on the rows.

#### Finding the Rollout Bottleneck

During the rollout, the main thread runs inference for one half of the envs while the worker threads step the physics of the other half. The console's `Phys Wait` line (and `Performance/Rollout_Physics_Wait` in TensorBoard, as a fraction) is the share of the rollout the main thread spent waiting for the workers:

- **High (about 25% and up):** physics is the limit. More worker threads help, up to the number of logical CPUs.
- **Low (about 10%):** the main thread's own work is the limit, so faster physics won't help much. On CPUs with spare cores, giving the main thread a core of its own can:

```cpp
// Logical CPU ids, as listed by `lscpu -e`. Empty (the default) leaves the threads unpinned.
config.main_thread_cpus = {0};                     // e.g. P-core 0 (CPUs 0 and 1 on hyperthreaded P-cores), CPU 1 left free.
config.worker_cpus = {2, 3, 4, /* ... */ 19};      // The workers share every other CPU.
config.num_threads = 0;                            // One worker per worker CPU.
```

On a 6 P-core + 8 E-core CPU this was about 5% faster, since the E-cores join the workers without competing with the main thread. On a 4-core CPU, where physics was the limit, reserving a core for the main thread was 6–10% slower, so measure before and after, or let [benchmark mode](#benchmarking-your-hardware) compare them. Startup fails if a listed CPU doesn't exist or is excluded by `taskset`, and warns if both lists share a CPU.

Phys Wait swings as the bot's play changes: a bot that keeps touching the ball makes physics the limit, and one that doesn't makes the main thread the limit. A setup from benchmark mode accounts for both. Training prints a `[WARN]`, once per run, when the median Phys Wait of the last 20 updates shows the CPU setup leaving speed unused:

- Physics is the limit (30% and up) while 2 or more CPUs have no worker thread.
- The main thread is the limit (10% and below) while it's unpinned and there are as many workers as CPUs, so it has no CPU of its own.

### Network

The actor-critic network is a shared MLP trunk that branches into separate actor and critic MLPs. Any of the three can be empty. For example, `shared_layers = {}` gives fully separate actor and critic networks.

```cpp
config.ac_cfg.shared_layers = { 1280, 1280, 1024 };
config.ac_cfg.actor_layers  = { 768, 512 };
config.ac_cfg.critic_layers = { 1024, 768 };
config.ac_cfg.use_layer_norm = true;
```

### PPO & Optimizer

```cpp
config.ppo_cfg.epochs = 1;
config.ppo_cfg.entropy_coef = 0.035f;
// Also available: clip_ratio, value_coef, max_grad_norm, gae_gamma, gae_lambda.

// Presets: OptimizerConfig::adam(lr), ::adamw(lr, wd), ::muon(muon_lr, adamw_lr, wd).
config.ppo_cfg.optimizer = OptimizerConfig::muon(3e-4, 3e-4, 0.01);
// Sections (shared / actor / critic) can be adjusted afterwards:
config.ppo_cfg.optimizer.critic.with_lr(1e-4f);
```

The Muon preset applies Muon to hidden-to-hidden weight matrices and AdamW to everything else. Its learning rate is scaled so values comparable to Adam's carry over.

### PFSP

PFSP is disabled by default (`envs = 0`). When enabled, a fraction of the envs in each even-team match distribution play against past versions of the learner instead of itself, and only the learner's side is trained on.

```cpp
config.pfsp_cfg.envs = 0.2f;     // Fraction of even-team envs that play a past version.
config.pfsp_cfg.versions = 2;    // Past versions played each update.
```

| Option | Default | Description |
|---|---|---|
| `envs` | `0.0` | Fraction of even-team envs playing past versions. `0` disables PFSP. |
| `versions` | `2` | Past versions sampled and played each update. Each env group needs at least this many PFSP envs. |
| `start_steps` | `0` | Until this step count, PFSP envs play a copy of the current learner. |
| `snapshot_interval` | `5'000'000` | Steps between snapshots of the learner added to the pool. |
| `pool_size` | `32` | Maximum snapshots kept. Past it, the one the learner beats most is removed. |
| `weighting` | `Variance` | `Variance` favors evenly matched versions. `Hard` favors versions the learner loses to. |
| `hard_exponent` | `2.0` | Exponent for `Hard` weighting. |
| `min_weight` | `0.02` | Minimum sampling weight, so no snapshot is ignored completely. |
| `win_rate_window` | `50` | Approximate number of recent games each win rate is estimated over. |

Only goals count toward win rates. Truncated episodes are ignored.

## Extending the Framework

**Rewards:** Subclass `reward::RewardFunction` and implement `get_reward(agent, current_state, previous_state)` and `clone()`. Optionally override `reset(initial_state)`, and `pre_tick(agents, current_state, previous_state)`, which runs once per tick before `get_reward` is called for any agent (useful for computing state shared across agents). Rewards are called once per agent per physics tick. Each env holds one instance shared by all of its agents, so store any per-agent state indexed by `agent.car_id`. Add a factory function to `src/reward/rewards.hpp`.

**Team Rewards:** For rewards that only depend on the team (goals, possession, ball toward goal), subclass `reward::TeamRewardFunction` and implement `get_team_reward(team, current_state, previous_state)` instead of `get_reward`. It is called once per team per tick, and every agent on that team receives the value. Team rewards are detected automatically. `self_share` has no effect on them, since teammates already receive the same value. With `opp_scale`, the opponent team's value is subtracted (not its total over its players), so `GoalReward` returning 1 to the scoring team with `opp_scale = 1` gives +1 and −1 in every match layout.

**Metrics:** Subclass `metric::MetricFunction` and implement `get_metric(agent, current_state, previous_state)` and `clone()`. Metrics are called once per agent per physics tick and return a `std::optional<double>`. Each value is added to a running sum and count, and the logged result is the sum divided by the count, so a metric returning 1 or 0 logs the fraction of ticks where its condition held. Returning `std::nullopt` leaves that tick out of both the sum and the count, which is how a metric can average only over the ticks it applies to (for example, ball speed only on ticks where the agent touched it). `get_metric` is `const`, so metrics can't keep state between ticks. Use `previous_state` for anything that compares against the last tick. Add a factory function to `src/metric/metrics.hpp`.

```cpp
class InAirRatioMetric : public metric::MetricFunction {
public:
    std::optional<double> get_metric(
        const env::AgentMeta& agent,
        const ffi::ArenaState& current_state,
        const ffi::ArenaState& /* previous_state */
    ) const override {
        // 1 when airborne, 0 on the ground. The logged average is the fraction of time spent in the air.
        return current_state.cars[agent.car_id].is_on_ground ? 0.0 : 1.0;
    }

    [[nodiscard]] std::unique_ptr<metric::MetricFunction> clone() const override {
        return std::make_unique<InAirRatioMetric>(*this);
    }
};
```

Only the agents being trained are measured. In PFSP envs, the past version's cars are left out. A metric that returns the same value for every agent, such as the ball's height, is still counted once per measured agent, so envs with more learner agents (for example 3v3 compared to 1v1) carry more weight in its average. Metrics are averaged per tick, so per-episode statistics (such as average episode length) aren't supported.

**State Setters:** Subclass `state::StateSetter` and implement `apply(arena, current_state, agents, rng)` and `clone()`. Add a factory function to `src/state/setters.hpp`.

New `.cpp` files must be added to the explicit source list in `CMakeLists.txt`.

**Observations & Actions:** The obs size and action space size are read from the builders, so the network adapts to changes automatically. The RLBot bot uses the obs and action code it was built with, so deploy a model with the code it was trained with (see [Deploying to RLBot](#deploying-to-rlbot)).

### Arena State Reference

Rewards, metrics, and state setters all read the game through `ffi::ArenaState` (declared in `include/buta_ppo/ffi.h`). It holds the world state after the current tick. It is never mirrored, so blue's goal is always at negative Y no matter which team an agent is on. Distances are in Unreal units (uu), velocities in uu/s, and angular velocities in rad/s. Index cars with `agent.car_id`.

| Field | Description |
| --- | --- |
| `ball.phys` | The ball's physics state (see `PhysState` below). |
| `cars[i].phys` | Car `i`'s physics state. |
| `cars[i].team` | `0` for blue, `1` for orange. |
| `cars[i].boost` | Boost amount, `0` to `100`. |
| `cars[i].is_on_ground`, `has_jumped`, `has_double_jumped`, `has_flipped`, `has_flip_or_jump`, `is_jumping`, `is_flipping`, `is_auto_flipping`, `is_supersonic` | Car movement flags. `has_flip_or_jump` is true while the car can still jump or flip. |
| `cars[i].air_time_since_jump`, `handbrake_val` | Seconds airborne since the last jump, and the handbrake amount. |
| `cars[i].is_demoed`, `demo_respawn_timer` | Whether the car is demolished, and the seconds until it respawns. |
| `boost_pads.big[6]`, `boost_pads.small[28]` | Seconds until each pad is available again. `0` means it can be picked up now. Pads are sorted by Y, then X. |
| `num_cars`, `tick_count` | Number of cars in the arena, and the arena's tick counter. |

**`PhysState`:** `pos[3]`, `vel[3]`, `ang_vel[3]`, and `rot_mat[3][3]`, where `rot_mat[0]` is the forward direction, `rot_mat[1]` is right, and `rot_mat[2]` is up. To work in angles, `util::math::euler_to_mat3` and `mat3_to_euler` (`src/util/math_utils.hpp`) convert between `rot_mat` and pitch, yaw and roll in radians, in the game's convention, the same as RLBot's `Rotator` and RLGym: positive pitch raises the nose, positive yaw turns toward the car's right (from +X toward +Y), and positive roll lowers its right side.

**Events (`events`):** Describe what happened during the current tick only.

| Field | Description |
| --- | --- |
| `is_ball_scored` | A goal was scored this tick. Check which goal from the ball's Y position. |
| `car_hit_ball[i]` | Car `i` touched the ball. |
| `ball_hit_extra_vel[i]` | Size of the extra velocity car `i`'s touch added to the ball. |
| `car_got_boost[i]` | Car `i` picked up a boost pad. |
| `car_bumped_car[b][v].bumped` | Car `b` bumped car `v`. Only real bumps count (the bumper drives into the victim, inside its bump cone), not every contact between cars. A head-on collision can register in both directions. |
| `car_bumped_car[b][v].is_demo` | The bump demolished car `v`. Only meaningful when `.bumped` is true. |

For example, a reward for bumping opponents:

```cpp
float get_reward(const env::AgentMeta& agent, const ffi::ArenaState& state, const ffi::ArenaState& /*prev_state*/) override {
    float reward = 0.0f;
    for (uint32_t v = 0; v < state.num_cars; v++) {
        const ffi::CarBump& bump = state.events.car_bumped_car[agent.car_id][v];
        if (bump.bumped && state.cars[v].team != state.cars[agent.car_id].team) {
            reward += bump.is_demo ? 1.0f : 0.5f;
        }
    }
    return reward;
}
```

## Checkpoints & the PFSP Pool

```text
checkpoints/
├── <bot_name>_<steps>/
│   ├── model.pt        # Network weights and the settings the bot needs to run them
│   └── optimizer.pt    # Optimizer state
└── <bot_name>_pool/    # PFSP opponent pool
    ├── <steps>.pt      # Snapshots, loadable and deployable like any model.pt
    ├── state.txt       # Win records
    └── meta.txt        # Network shape the pool was saved with
```

Checkpoints are written to a temporary directory and renamed when complete, so an interrupted save never produces a broken checkpoint. If the optimizer state can't be loaded (for example after switching optimizer presets), training continues with a fresh optimizer.

To roll back, delete the checkpoint directories newer than the one you want. On startup, pool snapshots newer than the resumed checkpoint are deleted automatically. If the network shape changed, the old pool is moved aside to `<bot_name>_pool.stale-<time>` and a fresh pool is started. TensorBoard logs are appended, not rolled back, so move the old event file out of `logs/` if you want clean curves.

## Telemetry

Metrics are written to `logs/<bot_name>.tfevents`, with lifetime steps on the x-axis.

```cpp
// Lifetime steps between TensorBoard logs, each aggregating the updates since the last. 0 logs every update.
config.steps_per_log = 0;
```

By default, every update is logged. With a larger `steps_per_log`, each logged point covers every update since the previous one: rewards and metrics are averaged over all of its steps and ticks, episode and PFSP counts are totals, the training statistics are averaged over its updates, and steps per second covers its whole duration. Log points land on multiples of `steps_per_log`, so they stay on schedule across resumes. A value below the actual steps per update logs every update. When training stops, any updates logged since the last point are written as one final, shorter point. The console output is still printed every update.

| Group | Metrics |
|---|---|
| `Reward_Components/` | Each reward's average weighted value per step, before the reward options. Shows how often the rewarded behavior happens |
| `Reward_Magnitude/` | Each reward's average absolute value per step, after the reward options. Shows how much each reward drives training |
| `Reward_Share/` | Each reward's fraction of the total magnitude, summing to 1. Shows which rewards dominate training. Not logged when no reward was received |
| `Reward/` | `Mean_Step` and `Mean_Abs_Step`, the totals of `Reward_Components/` and `Reward_Magnitude/` |
| `Episode/` | Goals, and truncations split into no-touch and time-limit, summed since the previous log |
| `Metrics/` | Each configured metric's average over every measured tick since the previous log. A metric that skipped every tick in that time isn't logged for it |
| `Loss/` | Policy, value, entropy |
| `Policy/` | Approx KL, clip fraction, normalized entropy |
| `Value/` | Explained variance (how much of the returns the critic predicts, independent of reward scale) |
| `PFSP/` | Win rate and games since the previous log, and pool size. The versions played each update are printed to the console |
| `Performance/` | Total and rollout steps per second, counted in learner agent steps and not including the time spent writing logs and checkpoints, and `Rollout_Physics_Wait`, the fraction of the rollout spent waiting on physics (see [Finding the Rollout Bottleneck](#finding-the-rollout-bottleneck)). `Env_Step_Microseconds` is the worker time per env step, and `Touch_Ticks_Per_Env_Step` and `Repredictions_Per_Env_Step` count the ball touches and full ball predictions that drive it, so a drop in SPS can be told apart from a bot that plays the ball more |

Rewards are averaged over learner agent steps, so PFSP opponents are left out. A reward that itself returns opposite values to the two teams (for example −1 to the team that conceded) averages to about 0 in `Reward_Components/` during self-play, since both teams are learners. Making it zero-sum with `opp_scale` instead keeps `Reward_Components/` meaningful, since it is logged before the options. `Reward_Magnitude/` doesn't cancel either way. The absolute value is taken per step, matching what PPO trains on, so a reward that flips sign within one step counts as their sum.

Approx KL and clip fraction are measured during each minibatch's forward pass, before its update. With `epochs = 1` and `num_minibatches = 1`, they only reflect small precision differences, not how far the update moved the policy.

Normalized entropy is each sample's entropy divided by its maximum, `ln(valid actions)`, averaged over the rollout. It runs from 0 (always the same action) to 1 (uniformly random over the actions the mask allows). The raw entropy also moves when the mix of masks changes, for example with more time in the air (about twice as many valid actions as on the ground) or more time without boost. Normalized entropy only moves when the policy itself becomes more or less random.

To view the logs with TensorBoard:

1. **Create a Virtual Environment** anywhere you like, and activate it. Activation has to be repeated in every new terminal, and the `activate` file depends on your shell.
```bash
python -m venv .venv
source .venv/bin/activate
```
2. **Install TensorBoard:**
```bash
pip install tensorboard
```
3. **Run TensorBoard:**
```bash
tensorboard --logdir=/path/to/logs --host 0.0.0.0 --port=6006
```

## Visualizer Mode

Setting `config.render = true;` in `main.cpp` skips training and runs a single environment in real time in the RocketSim visualizer, using the latest checkpoint for `bot_name`. By default the policy plays its most likely action, as the RLBot bot does, so the visualizer shows how the bot plays in a match. `config.render_best_action = false;` samples actions from the policy like training does instead.

### Playing Against Nexto

The visualizer can put Nexto, the RLGym community bot, on a team in place of your policy:

```cpp
auto nexto = create_nexto_bot("src/external/nexto/nexto-model.pt");
config.render_external_bots = {{nexto, ffi::Team::Orange}};  // Your bot plays blue.
```

List both teams for Nexto against itself. Nexto plays as the RLBot botpack version does, with its own observations, actions, decision timing, and scripted kickoff, and it plays through the 3 second countdown before each kickoff as it would in game (the visualizer doesn't wait for it). It runs on the GPU by default; `create_nexto_bot(path, {.device = torch::kCPU})` runs it on the CPU. Its options also set how random it plays (`beta`: `1`, the default, picks the best action, and `0.5` samples from its policy) and its kickoff behavior; see `NextoOptions` in `src/external/nexto/nexto_bot.hpp`.

External bots only work in the visualizer for now. Other bots can be added through the interface in `src/external/external_bot.hpp`. Nexto is included under the MIT License with permission from its author; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Deploying to RLBot

The `buta_ppo_bot` binary runs a trained model as an RLBot v5 bot. It builds the same observations as training, queries the model every `ticks_per_step` frames, and holds the chosen controls in between. Every saved model stores the network config (`ac_cfg`), `max_players_per_team`, and `ticks_per_step` it was trained with, and the bot sets itself up from those. Nothing has to be matched by hand, and one build of the bot runs any model.

1. **Build and Package:**
```bash
python deploy.py                                          # The latest checkpoint
python deploy.py checkpoints/<bot_name>_<steps>           # A specific checkpoint
python deploy.py checkpoints/<bot_name>_pool/<steps>.pt   # A PFSP pool snapshot
```
This builds the bot binary and copies it, the C++ inference library, the LibTorch libraries that library needs, the model (as `bot/model.pt`), and `assets/collision_meshes/` into `bot/`. When `checkpoints/` holds more than one bot, pick one with `--bot <bot_name>`. It needs the `LIBTORCH` environment variable set (see [Build Instructions](#build-instructions)). To switch models later without rebuilding, copy another `model.pt` or pool snapshot over `bot/model.pt`.

2. **Set Up the Bot Config:** The values marked `TEMPLATE` in `bot/bot.toml` and `bot/loadout.toml` belong to the original bot, so change them for yours. In particular `agent_id`, `name`, and `logo_file`, since the logo isn't included in the repository.
3. **Run:** Add the bot to a match through RLBot v5, using `bot/bot.toml`.

On startup the bot prints the settings it read from the model. It stops with an error if the model can't be run: a team in the match larger than its `max_players_per_team`, or an obs or action size that differs from what this build's `DefaultObs` and `DefaultAction` produce.

**Several Versions In-Game:** The bot runs the model named by the first argument of its run command, or `model.pt` without one. To play versions against each other, copy their models into `bot/` (e.g. `bot/models/<steps>.pt`) and make a copy of `bot/bot.toml` for each, with its own `agent_id`, `name`, and the model's path in both run commands, e.g. `run_command = "./buta_ppo_bot models/<steps>.pt"`.

**Obs and Action Code:** A model stores its settings, not the obs and action code. The bot uses the `DefaultObs` and `DefaultAction` it was built with, so if you change them after training, deploy older models from the code they were trained with. A change to the obs or action size stops the bot at startup, but other changes, such as a coefficient, aren't detected.

**Models Saved Before Configs Were Stored:** The bot and `deploy.py` refuse them. To re-save one, make its checkpoint the latest in `checkpoints/` and run training once with the update count in `runner.run(...)` set to `0`. That loads the checkpoint and saves it again in place, with the same weights and optimizer state plus the config. For a checkpoint that isn't the latest, move the newer ones out of `checkpoints/` first and set `pfsp_cfg.envs = 0` for that run, since resuming deletes pool snapshots newer than the resumed checkpoint.

**Running Elsewhere:** The bot runs inference on the CPU, on a single thread, so it uses neither the GPU nor more than one core while the game is running. With the demo network a decision took about 0.6 ms, and never more than 2.5 ms, on an i5-14600K with every core kept busy, well inside the 8.33 ms between game ticks. LibTorch's default multi-threaded CPU inference went past that window on every decision once all cores were busy, which is why the bot uses one thread. `bot/` carries the LibTorch libraries it needs (about 440 MB) in `bot/libs/` and finds them relative to itself, so the folder can be moved or copied to another x86-64 Linux machine without LibTorch, CUDA, or a GPU. That machine needs a glibc at least as new as the one the bot was built with, so a bot built on a rolling-release distro may need a recent distro to run.

## Versioning

buta-ppo follows [Semantic Versioning](https://semver.org) (`MAJOR.MINOR.PATCH`). The current version is **0.1.0**.

While the version is 0.x the framework is still settling, so:
* A **minor** release (0.1 → 0.2) may contain breaking changes.
* A **patch** release (0.1.0 → 0.1.1) only contains fixes and never breaks anything listed below.

A change counts as breaking if it:
* Makes existing checkpoints, optimizer state, or PFSP pools unloadable, or changes the observation or action layout so a trained model no longer fits (this includes the bot built by `deploy.py`).
* Renames, removes, or changes the meaning of a configuration field you set in `src/main.cpp`.
* Changes the `RewardFunction` or `StateSetter` interfaces so custom rewards or state setters need updating.

Breaking changes are called out in the release notes, so check them before updating mid-training.

## License

MIT. See [LICENSE](LICENSE). Includes third-party material under its own notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
