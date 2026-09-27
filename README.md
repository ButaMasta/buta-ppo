# buta-ppo

buta-ppo is a C++ reinforcement learning framework implementing Proximal Policy Optimization (PPO) for training Rocket League bots. Built using LibTorch and [RocketSim](https://crates.io/crates/rocketsim).

My main goal with this framework was to make use seamless and speeds respectable. Comparing against my own experience with [GigaLearn](https://github.com/ZealanL/GigaLearnCPP-Leak) I can confidently say it's at least twice as fast. As for training stability and quality though I cannot say as this project is so new I have not exstensively tested it.

## Architecture & Implementation Details

* **Memory Management:** The runtime loop operates with zero mid-step allocations. All environment states, observations, and action masks map directly to pre-allocated, pinned memory buffers initialized prior to training.
* **CUDA Graph Execution:** The forward pass, loss computation, and backpropagation phases of the PPO update loop are captured and replayed via CUDA graphs to eliminate host-to-device kernel launch latency.
* **Vectorization:** `VecEnv` executes environments in parallel across a thread pool, utilizing atomic counters and condition variables for thread synchronization and step batching.
* **FFI Bridge:** Simulation execution and state extraction are handled by a lightweight Rust Foreign Function Interface (FFI) bound directly to the RocketSim core.
* **Observation & Action Formatting:** Constructs agent observation tensors and applies X-axis state mirroring. Action masking maps discrete model outputs to simulator controls and filters invalid inputs based on the agent's current car state.

## Directory Structure

```text
buta-ppo/
├── assets/
│   └── collision_meshes/    # Required RocketSim arena geometry
├── checkpoints/             # Serialized LibTorch models (.pt)
├── include/buta_ppo/        # FFI definitions and headers
├── logs/                    # TensorBoard event telemetry
├── rust_bridge/             # Rust FFI wrapper for RocketSim
└── src/
    ├── action/              # Discrete action mapping
    ├── env/                 # Environment wrappers and state initialization
    ├── obs/                 # Observation vector generation and normalization
    ├── reward/              # Extensible reward functions and logging metrics
    ├── rl/                  # LibTorch PPO models, buffers, and training loops
    ├── util/                # Inline math and vector operations
    └── main.cpp             # Framework entry point and hyperparameter definition
```

## Prerequisites

* **Compiler:** C++20 compliant (GCC, Clang, or MSVC)
* **CMake:** Version 3.18 or higher
* **LibTorch:** CUDA-enabled C++ distribution
* **Rust Toolchain:** Cargo (required for the FFI bridge)
* **Collision Meshes:** Rocket League arena geometry placed in `assets/collision_meshes/`

## Build Instructions

The build process is automated via CMake, which will automatically fetch required C++ dependencies and compile the Rust FFI bridge during the build phase.

1. **Configure LibTorch Path:** Open `CMakeLists.txt` and update the `Torch_DIR` variable to point to the `share/cmake/Torch` directory of your local LibTorch installation.
```cmake
set(Torch_DIR "/path/to/your/libtorch/share/cmake/Torch")
```
2. **Make Necessary Directories:** Place your `collision_meshes/` directory in the `assets/` directory.
```bash
mkdir assets logs
```
3. **Build the Project:**
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
cd ..
```
4. **Subsequent Builds:** From the project root you can recompile using:
```bash
cmake --build build -j$(nproc)
```
5. **Execute the Framework:** Run the compiled binary from the root of the project to ensure the collision meshes are located correctly:
```bash
./build/buta_ppo
```


## Configuration

Hyperparameters, match types, and network topologies are strictly defined in `src/main.cpp` using the `RunnerConfig` struct.

### Match & Environment Distribution

The framework allocates active environment threads across defined match distributions and initial state setters.

```cpp
// Allocate environments based on relative weights
config.match_distributions = {
    {1, 1, 1.0f} // 1v1 Matches
};

// Define initial physical states
config.setter_distributions = {
    {DefaultKickoffSetter(), 0.40f},              // Standard kickoff
    {RandomStateSetter(true, true, true), 0.30f}, // Randomized bounds (Grounded)
    {RandomStateSetter(true, true, false), 0.30f} // Randomized bounds (Aerial)
};
```

### Network Topology

The Actor-Critic implementation supports a shared multilayer perceptron (MLP) feature extractor that branches into independent actor and critic heads.

```cpp
config.ac_cfg.shared_layers = { 1280, 1280, 1024 };
config.ac_cfg.actor_layers  = { 768, 512 };
config.ac_cfg.critic_layers = { 1024, 768 };
config.ac_cfg.use_layer_norm = true;
```

## Execution & Telemetry

Executing the compiled binary will automatically search the `checkpoints/` directory for the most recent `.pt` weights matching the configured `bot_name`. If found, training resumes from that global step; otherwise, a fresh model is initialized.

To safely interrupt training, send a `SIGINT` (Ctrl+C). The framework will catch the signal, complete the active optimization epoch, serialize the model state to disk, and exit.

### TensorBoard Logging:
Metrics including policy loss, value loss, entropy, and individual reward component averages are written to `logs/`. In order to setup tensorboard follow these steps:

1. **Choose A Location For Your Virtual Environment:** This can be anywhere you'd like.
2. **Create The Virtual Environment:**
```bash
python -m venv .venv
```
3. **Activate The Environment:** This will have to be done anytime you want to view the tensorboard if the environment is not already active. Note that the specific `activate` file you select depends on your terminal.
```bash
source .venv/bin/activate
```
4. **Install TensorBoard:**
```bash
pip install tensorboard
```
5. **Run TensorBoard:**
```bash
tensorboard --logdir=/path/to/logs --host 0.0.0.0 --port=6006
```

### Visualizer Mode:
Setting `config.render = true;` in `main.cpp` will bypass the optimization loop, restrict execution to a single environment thread, and launch the RocketSim visualizer for real-time policy evaluation.
