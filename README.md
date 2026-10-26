# buta-ppo

buta-ppo is a C++ reinforcement learning framework implementing Proximal Policy Optimization (PPO) for training Rocket League bots. Built using LibTorch and [RocketSim](https://crates.io/crates/rocketsim).

My main goal with this framework was to make use seamless and speeds respectable. Comparing against my own experience with [GigaLearn](https://github.com/ZealanL/GigaLearnCPP-Leak) I can confidently say it's at least four times as fast. As for training stability and quality though I cannot say as this project is so new I have not extensively tested it though there are some bots which have surpassed nexto level utilizing this framework already.

For reference, the demo configuration in `src/main.cpp` (1v1, 2v2, and 3v3, 500 envs, a ~6M parameter network, PFSP enabled, unpinned) trains at about 580k steps per second with a fresh policy, and about 550k once the bot begins to properly interact with the ball, on an i5-14600K with an RTX 5070. Steps here are agent steps: each learner car's step counts once, so a 1v1 env step is two steps, or one when a PFSP opponent drives the other car. The console's `Total SPS` and TensorBoard's `Performance/Total_SPS` count the same way.

## Features

* **PPO:** GAE advantages, clipped surrogate objective, entropy bonus, advantage normalization over the whole rollout, and global gradient clipping.
* **Pipelined Rollout:** Environments are split into two groups. While worker threads step one group's physics, the main thread runs inference for the other, so the CPU and GPU stay busy at the same time.
* **CUDA Graphs:** Rollout inference, each rollout step's writes into the buffer, and the entire training minibatch step (forward, loss, backward, gradient clipping, optimizer step) are captured once and replayed, removing kernel launch overhead. Training runs under bf16 autocast, and rollout inference uses TF32 matmuls.
* **Zero Mid-Step Allocations:** Observations, action masks, rewards, and actions live in persistent, pinned buffers allocated before training starts.
* **Graph-Safe Optimizers:** Adam, AdamW, and Muon (on hidden weight matrices), reimplemented so they can run inside a CUDA graph. They can be configured per network section and parameter role.
* **PFSP:** Prioritized fictitious self-play pits a fraction of the environments against past versions of the learner, chosen by the learner's win rate against them.
* **Observations & Actions:** Team-relative observations with X-axis mirroring, ball prediction, and shuffled car slots. A 90-action discrete lookup table with action masking based on the car's state.
* **Rust FFI Bridge:** Physics runs in RocketSim through a lightweight C ABI exported from Rust.
* **Skill Tracking:** Periodic evaluation games against fixed past versions of the bot and against Nexto, the RLGym community bot, give a Bradley–Terry Elo rating per team size on a fixed scale (Nexto = 1500), so you can see whether the bot is actually getting better, and how far it is from Nexto.
* **Telemetry & Checkpoints:** TensorBoard logging with custom per-tick metrics, automatic resume from the latest checkpoint, and a persistent PFSP opponent pool.
* **RLBot Deployment:** A Rust RLBot v5 bot binary that runs trained checkpoints in real matches.

## How It Works

Each update runs through the following stages:

1. **Rollout:** `VecEnv` steps every `RocketSimEnv` on a thread pool. Each env applies its agents' actions, ticks the arena `ticks_per_step` times while accumulating rewards and metrics, and builds the next observations and action masks. Finished episodes are reset automatically.
2. **Inference:** Each env group's policy forward pass and action sampling run as a single CUDA graph replay that reads directly from `VecEnv`'s batched buffers.
3. **Advantages:** `RolloutBuffer` stores every step on the GPU, then computes GAE advantages and returns. Truncated episodes are bootstrapped with the critic's estimate of the final state.
4. **Training:** `PPOTrainer` shuffles the rollout into minibatches and replays the captured training graph for each one.
5. **Checkpointing:** When training ends or is interrupted, the model and optimizer state are saved, and they are loaded automatically on the next start.
6. **Evaluation:** Every `steps_per_eval` steps, training pauses while the learner plays a fixed set of games against past versions of itself and against Nexto, and its rating is logged (see [Evaluation](docs/evaluation.md)).

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
To use a specific release instead, clone its tag, e.g. `git clone -b v0.3.0 https://github.com/ButaMasta/buta-ppo.git`.

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

Resuming, stopping, and checkpoints are covered in [Training](docs/training.md), and every setting in [Configuration](docs/configuration.md).

## Documentation

| Page | Covers |
|---|---|
| [Training](docs/training.md) | Resuming and stopping, the update cap, checkpoints, the PFSP pool, and rolling back |
| [Configuration](docs/configuration.md) | Matches, state setters, episode limits, rewards and reward options, metrics, envs and batch size, the network, and PPO and the optimizer |
| [Performance](docs/performance.md) | Benchmark mode, the rollout bottleneck (`Phys Wait`), and CPU pinning |
| [PFSP](docs/pfsp.md) | Self-play against past versions of the learner, and its options |
| [Evaluation](docs/evaluation.md) | Elo ratings, Nexto and its ladder, anchors, options, reading the curve, and the cost |
| [Telemetry](docs/telemetry.md) | TensorBoard groups and tags, and how to view them |
| [Visualizer Mode](docs/visualizer.md) | Watching the bot in the RocketSim visualizer, and playing it against Nexto |
| [Deploying to RLBot](docs/deploying.md) | Packaging the bot and running it in RLBot v5 |
| [Extending the Framework](docs/extending.md) | Custom rewards, metrics, and state setters, the arena state reference, and the directory structure |

## Versioning

buta-ppo follows [Semantic Versioning](https://semver.org) (`MAJOR.MINOR.PATCH`). The current version is **0.3.0**.

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
