# Deploying to RLBot

The `buta_ppo_bot` binary runs a trained model as an RLBot v5 bot. It builds the same observations as training, queries the model every `ticks_per_step` frames, and holds the chosen controls in between. Every saved model stores the network config (`ac_cfg`), `max_players_per_team`, and `ticks_per_step` it was trained with, and the bot sets itself up from those. Nothing has to be matched by hand, and one build of the bot runs any model.

1. **Build and Package:**
```bash
python deploy.py                                          # The latest checkpoint
python deploy.py checkpoints/<bot_name>_<steps>           # A specific checkpoint
python deploy.py checkpoints/<bot_name>_pool/<steps>.pt   # A PFSP pool snapshot
```
This builds the bot binary and copies it, the C++ inference library, the LibTorch libraries that library needs, the model (as `bot/model.pt`), and `assets/collision_meshes/` into `bot/`. When `checkpoints/` holds more than one bot, pick one with `--bot <bot_name>`. It needs the `LIBTORCH` environment variable set (see [Build Instructions](../README.md#build-instructions)). To switch models later without rebuilding, copy another `model.pt` or pool snapshot over `bot/model.pt`.

2. **Set Up the Bot Config:** The values marked `TEMPLATE` in `bot/bot.toml` and `bot/loadout.toml` belong to the original bot, so change them for yours. In particular `agent_id`, `name`, and `logo_file`, since the logo isn't included in the repository.
3. **Run:** Add the bot to a match through RLBot v5, using `bot/bot.toml`.

On startup the bot prints the settings it read from the model. It stops with an error if the model can't be run: a team in the match larger than its `max_players_per_team`, or an obs or action size that differs from what this build's `DefaultObs` and `DefaultAction` produce.

**Several Versions In-Game:** The bot runs the model named by the first argument of its run command, or `model.pt` without one. To play versions against each other, copy their models into `bot/` (e.g. `bot/models/<steps>.pt`) and make a copy of `bot/bot.toml` for each, with its own `agent_id`, `name`, and the model's path in both run commands, e.g. `run_command = "./buta_ppo_bot models/<steps>.pt"`.

**Obs and Action Code:** A model stores its settings, not the obs and action code. The bot uses the `DefaultObs` and `DefaultAction` it was built with, so if you change them after training, deploy older models from the code they were trained with. A change to the obs or action size stops the bot at startup, but other changes, such as a coefficient, aren't detected.

**Models Saved Before Configs Were Stored:** The bot and `deploy.py` refuse them. To re-save one, make its checkpoint the latest in `checkpoints/` and run training once with the update count in `runner.run(...)` set to `0`. That loads the checkpoint and saves it again in place, with the same weights and optimizer state plus the config. For a checkpoint that isn't the latest, move the newer ones out of `checkpoints/` first and set `pfsp_cfg.envs = 0` for that run, since resuming deletes pool snapshots newer than the resumed checkpoint.

**Running Elsewhere:** The bot runs inference on the CPU, on a single thread, so it uses neither the GPU nor more than one core while the game is running. With the demo network a decision took about 0.6 ms, and never more than 2.5 ms, on an i5-14600K with every core kept busy, well inside the 8.33 ms between game ticks. LibTorch's default multi-threaded CPU inference went past that window on every decision once all cores were busy, which is why the bot uses one thread. `bot/` carries the LibTorch libraries it needs (about 440 MB) in `bot/libs/` and finds them relative to itself, so the folder can be moved or copied to another x86-64 Linux machine without LibTorch, CUDA, or a GPU. That machine needs a glibc at least as new as the one the bot was built with, so a bot built on a rolling-release distro may need a recent distro to run.
