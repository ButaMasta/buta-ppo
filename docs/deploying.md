# Deploying to RLBot

The `buta_ppo_bot` binary runs a trained model as an RLBot v5 bot. It builds the same observations as training, queries the model every `ticks_per_step` frames, and holds the chosen controls in between. Every saved model stores the network config (`ac_cfg`), `max_players_per_team`, and `ticks_per_step` it was trained with, and the bot sets itself up from those. Nothing has to be matched by hand, and one build of the bot runs any model.

1. **Build and Package:**
```bash
python deploy.py                                          # The latest checkpoint
python deploy.py checkpoints/<bot_name>_<steps>           # A specific checkpoint
python deploy.py checkpoints/<bot_name>_pool/<steps>.pt   # A PFSP pool snapshot
```
This builds the bot for the OS you run it on and copies the bot binary, the C++ inference library, the Rust bridge library, the LibTorch libraries those need (into `bot/libs/`), the model (as `bot/model.pt`), and `assets/collision_meshes/` into `bot/`. When `checkpoints/` holds more than one bot, pick one with `--bot <bot_name>`. It needs the `LIBTORCH` environment variable set (see [Build Instructions](../README.md#build-instructions)). `--test` runs the bot's [self-test](#self-test) afterwards. To build a Windows bot, see [Windows](#windows). To switch models later without rebuilding, copy another `model.pt` or pool snapshot over `bot/model.pt`.

2. **Set Up the Bot Config:** The values marked `TEMPLATE` in `bot/bot.toml` and `bot/loadout.toml` belong to the original bot, so change them for yours. In particular `agent_id`, `name`, and `logo_file`, since the logo isn't included in the repository.
3. **Run:** Add the bot to a match through RLBot v5, using `bot/bot.toml`.

On startup the bot prints the settings it read from the model. It stops with an error if the model can't be run: a team in the match larger than its `max_players_per_team`, or an obs or action size that differs from what this build's `DefaultObs` and `DefaultAction` produce. During a match it plays any layout up to that size, empty teams included, and picks up players who join after it started. While it can't play (a team grows past `max_players_per_team`, or the game state can't be mirrored in RocketSim) it holds neutral controls and prints why, and plays again once it can.

**Several Versions In-Game:** The bot runs the model named by the first argument of its run command, or `model.pt` without one. To play versions against each other, copy their models into `bot/` (e.g. `bot/models/<steps>.pt`) and make a copy of `bot/bot.toml` for each, with its own `agent_id`, `name`, and the model's path in both run commands, e.g. `run_command = ".\\buta_ppo_bot.exe models\\<steps>.pt"` (Windows) and `run_command_linux = "./buta_ppo_bot models/<steps>.pt"`.

**Obs and Action Code:** A model stores its settings, not the obs and action code. The bot uses the `DefaultObs` and `DefaultAction` it was built with, so if you change them after training, deploy older models from the code they were trained with. A change to the obs or action size stops the bot at startup, but other changes, such as a coefficient, aren't detected.

The C++ library links its own copy of the Rust bridge, as training does, so obs code can call any function in `rust_bridge/src/lib.rs` (ball prediction, a RocketSim arena of its own, or one you add) and the bot builds without further changes. The bot binary has a separate copy for mirroring the game. The library loads the collision meshes for its copy at startup, which takes about 13 ms and 5 MiB.

**Building by Hand:** `deploy.py` runs `cargo build --release -p rust_bridge` before `cargo build --release -p buta_ppo_bot` (both in `rust_bridge/`, plus `--target x86_64-pc-windows-msvc` for Windows). The bridge has to come first, since the C++ library links it and Cargo would build it alongside the bot's build script rather than before it. The bot build stops with the command to run if the bridge library is missing or older than its source. `ninja -C build` also builds the Linux bridge.

**Models Saved Before Configs Were Stored:** The bot and `deploy.py` refuse them. To re-save one, make its checkpoint the latest in `checkpoints/` and run training once with the update count in `runner.run(...)` set to `0`. That loads the checkpoint and saves it again in place, with the same weights and optimizer state plus the config. For a checkpoint that isn't the latest, move the newer ones out of `checkpoints/` first and set `pfsp_cfg.envs = 0` for that run, since resuming deletes pool snapshots newer than the resumed checkpoint.

## Windows

RLBot and Rocket League run on Windows, so the bot can be built for it, on Windows or from Linux. Training needs Linux, but the bot doesn't: copy a `model.pt` from a Linux checkpoint. `bot/` holds one OS's build at a time, since `bot/libs/` does, and `bot/bot.toml` has a run command for each.

**On Windows:** Install Visual Studio (or its Build Tools) with the C++ workload, Rust through rustup (its default Windows toolchain is the MSVC one), CMake, Python 3, and the Windows CPU LibTorch (`https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-<version>%2Bcpu.zip`, with `<version>` the LibTorch version you train with). Set `LIBTORCH` (or `LIBTORCH_WINDOWS`) to it, put the collision meshes in `assets/`, and run `python deploy.py <model>`.

**From Linux:** `deploy.py --os windows` cross-compiles the bot with [cargo-xwin](https://github.com/rust-cross/cargo-xwin), which builds with clang-cl and lld-link against Microsoft's C runtime and Windows SDK. It needs:
```bash
rustup target add x86_64-pc-windows-msvc
cargo install --locked cargo-xwin
# clang, lld and llvm from your package manager
export LIBTORCH_WINDOWS=/path/to/libtorch-windows   # the Windows CPU LibTorch, as above
python deploy.py --os windows --accept-msvc-license --test
```
cargo-xwin downloads Microsoft's C runtime and Windows SDK on the first build, under [Microsoft's license terms](https://go.microsoft.com/fwlink/?LinkId=2086102), so the first run needs `--accept-msvc-license`. `--test` runs the self-test with Wine, if it's installed. Copy `bot/` to the Windows machine afterwards.

**Libraries:** `bot/libs/` gets the DLLs the bot's libraries import from LibTorch (followed through their own imports) and Microsoft's Visual C++ runtime DLLs, so the bot runs on Windows 10 or newer without installing anything. `deploy.py` takes the runtime from Microsoft's latest installer (`https://aka.ms/vc14/vc_redist.x64.exe`), downloaded once into `rust_bridge/target/msvc-runtime/`; set `MSVC_RUNTIME_DIR` to a folder of x64 runtime DLLs to use those instead, e.g. Visual Studio's `VC\Redist\MSVC\<version>\x64\Microsoft.VC14x.CRT`. Windows has no rpath, so `buta_ppo_bot.exe` loads `libs\buta_ppo_bot.dll` itself when it starts, and exits with an error saying a library is missing from `libs\` if that or a DLL it needs can't be found.

## Self-Test

`bot/buta_ppo_bot --self-test [model]` (`buta_ppo_bot.exe` on Windows) checks the bot runs on this machine without RLBot or the game. It loads the model as a match would, plays 8 decisions from a seeded RocketSim kickoff in each team size the model supports, and prints every car's controls. It runs the given model, else `bot/model.pt`, else a small 1v1 test model with seeded random weights that it writes for the current obs and action code. Run it from `bot/`, since it needs `collision_meshes/` there. A 1v1 model's output is the same on every run and on Linux and Windows, which [CI](#continuous-integration) checks; with more car slots than cars it varies, because the obs builder shuffles the slots with a random seed.

## Continuous Integration

`.github/workflows/bot.yml` builds the bot with `deploy.py` for Linux, for Windows on Windows, and for Windows from Linux, runs each build's self-test on its OS, and checks the three outputs match. It doesn't build training, which needs CUDA.

**It's off by default.** GitHub Actions only runs for a copy of the repository hosted on GitHub (a fork, or your clone pushed to a repository of your own), and even there the jobs are skipped unless the repository's Actions variable `BOT_CI` is `true`. A skipped run still shows in the Actions tab, but uses no minutes. To turn it on, set the variable under Settings → Secrets and variables → Actions → Variables, or with `gh variable set BOT_CI --body true`. To run it once without turning it on, start "Bot builds" by hand from the Actions tab. When on, it runs for pushes to `main` and for pull requests that change the bot's code (`src/` except `src/main.cpp`, `include/`, `rust_bridge/`, `CMakeLists.txt`, `deploy.py`, or the workflow). One run takes about 20 minutes of a private repository's monthly Actions minutes, since GitHub counts Windows minutes double; public repositories don't pay for minutes. The Windows bot cross-built from Linux is kept as the `bot-windows-cross` artifact of each run, without `model.pt` and `collision_meshes/`.

The self-tests need the collision meshes, which come from the `COLLISION_MESHES` repository secret (see `.github/workflows/collision_meshes.py` for how to make it). Without it, as in pull requests from forks, the builds still run and the tests are skipped.

**Locally,** without GitHub: `python deploy.py --test` builds the bot for your OS and runs its self-test, and on Linux `python deploy.py --os windows --test` does the same for a Windows build under Wine.

## Running Elsewhere

The bot runs inference on the CPU, on a single thread, so it uses neither the GPU nor more than one core while the game is running. With the demo network a decision took about 0.6 ms, and never more than 2.5 ms, on an i5-14600K with every core kept busy, well inside the 8.33 ms between game ticks. LibTorch's default multi-threaded CPU inference went past that window on every decision once all cores were busy, which is why the bot uses one thread. `bot/` carries the LibTorch libraries it needs in `bot/libs/` (about 440 MB on Linux, 300 MB on Windows) and finds them relative to itself, so the folder can be moved or copied to another x86-64 machine of the same OS without LibTorch, CUDA, or a GPU. On Linux, that machine needs a glibc at least as new as the one the bot was built with, so a bot built on a rolling-release distro may need a recent distro to run.
