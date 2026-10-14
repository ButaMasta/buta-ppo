# buta-ppo

buta-ppo is a C++ reinforcement learning framework implementing Proximal Policy Optimization (PPO) for training Rocket League bots. Built using LibTorch and [RocketSim](https://crates.io/crates/rocketsim).

My main goal with this framework was to make use seamless and speeds respectable. Comparing against my own experience with [GigaLearn](https://github.com/ZealanL/GigaLearnCPP-Leak) I can confidently say it's at least twice as fast. As for training stability and quality though I cannot say as this project is so new I have not extensively tested it.

For reference, the demo configuration in `src/main.cpp` (1v1, 253 envs, a ~6M parameter network, PFSP enabled) trains at roughly 300k steps per second on an i5-14600K with an RTX 5070.

## Features

* **PPO:** GAE advantages, clipped surrogate objective, entropy bonus, per-batch advantage normalization, and global gradient clipping.
* **Pipelined Rollout:** Environments are split into two groups. While worker threads step one group's physics, the main thread runs inference for the other, so the CPU and GPU stay busy at the same time.
* **CUDA Graphs:** Rollout inference and the entire training minibatch step (forward, loss, backward, gradient clipping, optimizer step) are captured once and replayed, removing kernel launch overhead. Training runs under bf16 autocast, and rollout inference uses TF32 matmuls.
* **Zero Mid-Step Allocations:** Observations, action masks, rewards, and actions live in persistent, pinned buffers allocated before training starts.
* **Graph-Safe Optimizers:** Adam, AdamW, and Muon (on hidden weight matrices), reimplemented so they can run inside a CUDA graph. They can be configured per network section and parameter role.
* **PFSP:** Prioritized fictitious self-play pits a fraction of the environments against past versions of the learner, chosen by the learner's win rate against them.
* **Observations & Actions:** Team-relative observations with X-axis mirroring, ball prediction, and shuffled car slots. A 90-action discrete lookup table with action masking based on the car's state.
* **Rust FFI Bridge:** Physics runs in RocketSim through a lightweight C ABI exported from Rust.
* **Telemetry & Checkpoints:** TensorBoard logging, automatic resume from the latest checkpoint, and a persistent PFSP opponent pool.
* **RLBot Deployment:** A Rust RLBot v5 bot binary that runs trained checkpoints in real matches.

## How It Works

Each update runs through the following stages:

1. **Rollout:** `VecEnv` steps every `RocketSimEnv` on a thread pool. Each env applies its agents' actions, ticks the arena `ticks_per_step` times while accumulating rewards, and builds the next observations and action masks. Finished episodes are reset automatically.
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
│   ├── obs/                 # Observation builder and ball prediction
│   ├── reward/              # Reward functions and the reward manager
│   ├── rl/                  # Network, rollout buffer, trainer, optimizers, PFSP, and the training runner
│   ├── state/               # State setters for episode resets
│   ├── util/                # Inline math and vector operations
│   ├── bot_api.cpp          # Inference-only library used by the RLBot bot
│   └── main.cpp             # Entry point and all training configuration
└── deploy.py                # Builds and packages the RLBot bot
```

## Prerequisites

* **OS & Compiler:** Linux with a C++20 compiler (GCC or Clang).
* **GPU:** An NVIDIA GPU with CUDA. CPU training is not supported. Training runs in bf16, so an Ampere (RTX 30-series) or newer GPU is recommended. Older GPUs are untested.
* **LibTorch:** The CUDA-enabled C++ distribution. The trainer relies on recent LibTorch internals (fused optimizer kernels, CUDA graphs), so older versions may not build.
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

1. **Configure the LibTorch Path:** The LibTorch location is hardcoded in three places. Update all of them to point to your installation:
    * `CMakeLists.txt`: `set(Torch_DIR "/path/to/libtorch/share/cmake/Torch")`
    * `rust_bridge/build.rs`: the `Torch_DIR` define and `torch_dir`
    * `deploy.py`: `torch_lib_dir`
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

## Running Training

Run the compiled binary from the project root, so the relative paths to `assets/`, `checkpoints/`, and `logs/` resolve:
```bash
./build/buta_ppo
```

On startup, the framework searches `checkpoints/` for the most recent checkpoint matching the configured `bot_name`. If one is found, training resumes from its step count. Otherwise, a fresh model is initialized.

A run trains for at most the number of updates passed to `runner.run` in `main.cpp` (`1'000` by default, about 50M steps with the demo config), then saves a checkpoint and exits. Run the binary again to continue from there.

To stop training safely, press Ctrl+C. The current update finishes, a checkpoint is saved, and the program exits.

Checkpoints are also saved every `config.steps_per_checkpoint` lifetime steps (`100'000'000` by default; `0` turns this off), so a crash only loses the progress since the last one. Every checkpoint is kept, and each one is roughly as large as the network and its optimizer state (about 50 MB for the demo network), so delete old ones you don't need.

## Configuration

All configuration lives in `src/main.cpp`, in the `RunnerConfig` struct. The values there are demo values that show the available options, not tuned hyperparameters.

### Matches, State Setters & Rewards

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

// Rewards with their names (for logging) and weights. Rewards are computed every physics tick.
config.reward_entries = {
    {"VelocityToBall", create_velocity_to_ball_reward(), 0.1f},
    {"TouchBall", create_touch_ball_reward(), 1.0f},
    {"Goal", create_goal_reward(), 10.0f}
};
```

Rewards are summed over the `ticks_per_step` ticks of each step, so per-tick rewards scale with `ticks_per_step`.

### Environments & Batch Size

```cpp
config.num_threads = 11;            // Worker threads stepping the envs.
config.num_envs_per_thread = 23;    // Total envs = threads * envs per thread. Training needs at least 2.
config.ticks_per_step = 8;          // Physics ticks (120 per second) each action is held for.

config.target_steps_per_update = 50'000;
config.num_minibatches = 1;
```

The actual steps per update are rounded up from `target_steps_per_update`, so every agent contributes the same number of steps and the total divides evenly into `num_minibatches`. The update size is limited by VRAM. Start with a small `target_steps_per_update` or more minibatches, and increase from there.

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

**Rewards:** Subclass `reward::RewardFunction` and implement `get_reward(agent, current_state, previous_state)` and `clone()`. Optionally override `reset(initial_state)`. Rewards are called once per agent per physics tick. Each env holds one instance shared by all of its agents, so store any per-agent state indexed by `agent.car_id`. Add a factory function to `src/reward/rewards.hpp`.

**State Setters:** Subclass `state::StateSetter` and implement `apply(arena, current_state, agents, rng)` and `clone()`. Add a factory function to `src/state/setters.hpp`.

New `.cpp` files must be added to the explicit source list in `CMakeLists.txt`.

**Observations & Actions:** The obs size and action space size are read from the builders, so the network adapts to changes automatically. `src/bot_api.cpp` has to be kept in sync by hand (see [Deploying to RLBot](#deploying-to-rlbot)).

## Checkpoints & the PFSP Pool

```text
checkpoints/
├── <bot_name>_<steps>/
│   ├── model.pt        # Network weights
│   └── optimizer.pt    # Optimizer state
└── <bot_name>_pool/    # PFSP opponent pool
    ├── <steps>.pt      # Snapshots, loadable like any model.pt
    ├── state.txt       # Win records
    └── meta.txt        # Network shape the pool was saved with
```

Checkpoints are written to a temporary directory and renamed when complete, so an interrupted save never produces a broken checkpoint. If the optimizer state can't be loaded (for example after changing the network), training continues with a fresh optimizer.

To roll back, delete the checkpoint directories newer than the one you want. On startup, pool snapshots newer than the resumed checkpoint are deleted automatically. If the network shape changed, the old pool is moved aside to `<bot_name>_pool.stale-<time>` and a fresh pool is started. TensorBoard logs are appended, not rolled back, so move the old event file out of `logs/` if you want clean curves.

## Telemetry

Metrics are written to `logs/<bot_name>.tfevents`, with lifetime steps on the x-axis.

| Group | Metrics |
|---|---|
| `Reward_Components/`, `Reward/` | Each reward's average per step, and the total |
| `Episode/` | Goals, and truncations split into no-touch and time-limit |
| `Loss/` | Policy, value, entropy |
| `Policy/` | Approx KL, clip fraction |
| `Value/` | Explained variance (how much of the returns the critic predicts, independent of reward scale) |
| `PFSP/` | Win rate overall and per slot, the versions played, pool size, games |
| `Performance/` | Total and rollout steps per second |

Approx KL and clip fraction are measured during each minibatch's forward pass, before its update. With `epochs = 1` and `num_minibatches = 1`, they only reflect small precision differences, not how far the update moved the policy.

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

Setting `config.render = true;` in `main.cpp` skips training and runs a single environment in real time in the RocketSim visualizer, using the latest checkpoint for `bot_name`. The visualizer samples actions from the policy like training does. The RLBot bot always picks the most likely action, so it can behave differently from what the visualizer shows.

## Deploying to RLBot

The `buta_ppo_bot` binary runs a trained checkpoint as an RLBot v5 bot. It builds the same observations as training, queries the model every `tick_skip` frames, and holds the chosen controls in between.

1. **Match the Training Config:** `src/bot_api.cpp` hardcodes its own settings, which must match the config the checkpoint was trained with:
    * `MAX_PLAYERS_PER_TEAM` = `max_players_per_team`
    * `shared_layers`, `actor_layers`, `critic_layers`, `use_layer_norm` = the `ac_cfg` values
    * The tick skip passed to `DefaultObs`, and `tick_skip` in `rust_bridge/src/main.rs` = `ticks_per_step`

    A wrong layer size fails at startup. A wrong tick skip doesn't fail, it just plays worse.
2. **Build and Package:**
```bash
python deploy.py
```
This builds the bot binary and copies it, the C++ inference library, and the main LibTorch libraries into `bot/`.

3. **Add the Remaining Files:** Copy `checkpoints/<bot_name>_<steps>/model.pt` and `assets/collision_meshes/` into `bot/`. The values marked `TEMPLATE` in `bot/bot.toml` and `bot/loadout.toml` belong to the original bot, so change them for yours. In particular `agent_id`, `name`, and `logo_file`, since the logo isn't included in the repository.
4. **Run:** Add the bot to a match through RLBot v5, using `bot/bot.toml`.

The packaged bot still loads some CUDA and LibTorch libraries from your local LibTorch installation, so it will only run on a machine with LibTorch installed at the same path.

## License

MIT. See [LICENSE](LICENSE).
