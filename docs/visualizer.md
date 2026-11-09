# Visualizer Mode

Setting `config.render = true;` in `main.cpp` skips training and runs a single environment in real time in the RocketSim visualizer, using the latest checkpoint for `bot_name`. By default the policy plays its most likely action, as the RLBot bot does, so the visualizer shows how the bot plays in a match. `config.render_best_action = false;` samples actions from the policy like training does instead.

The visualizer is a separate app, `rocketsim-viewer`, which the framework starts (or connects to, if it's already open) when render mode begins. Install the version matching the framework's RocketSim once, either as a prebuilt binary with [cargo-binstall](https://github.com/cargo-bins/cargo-binstall), which takes seconds:

```bash
cargo binstall rocketsim_viewer@0.3.3
```

or by building it from source, which takes a few minutes:

```bash
cargo install --locked rocketsim_viewer --version 0.3.3
```

It needs to be on your `PATH` (Cargo installs to `~/.cargo/bin`), or set the `ROCKETSIM_VIEWER` environment variable to its path. Without it, render mode stops at startup with an error saying so. The viewer has its own camera controls; its car controls aren't used, since the policy and any external bots drive every car.

## Viewer Keys

While the viewer follows a car, two of its keys restart the episode:

| Key | Does |
|---|---|
| `Backspace` (or the settings menu's Reset play) | Starts a new episode from the state setters, as a goal or timeout would. |
| `F4` | Replays the start of the current episode: the same setter and the same state again. |

The viewer only sends these while it follows a car (`1` to `8`, or `C` to cycle through the cars and the overhead view), has focus and has its settings menu closed, and not in its spectate-only mode. Following a car only moves the camera here; it doesn't hand you the car. `F2` (the viewer's dribble key) does nothing. A replay starts from exactly the same state, but the observation's mirroring and car order are drawn anew, as at every reset, so the bots can play it differently.

To look at particular state setters, such as a new one, list them in `render_setter_distributions`. Render then starts every episode from them instead of `setter_distributions`, and training is unaffected:

```cpp
config.render_setter_distributions = {{create_default_kickoff_setter(), 1.0f}};
```

Render plays one match layout, the first among those with the largest weight in `match_distributions`.

## Playing Against Nexto

The visualizer can put Nexto, the RLGym community bot, on a team in place of your policy:

```cpp
auto nexto = create_nexto_bot("src/external/nexto/nexto-model.pt");
config.render_external_bots = {{nexto, ffi::Team::Orange}};  // Your bot plays blue.
```

List both teams for Nexto against itself. Nexto plays as the RLBot botpack version does, with its own observations, actions, decision timing, and scripted kickoff, and it plays through the 3 second countdown before each kickoff as it would in game (the visualizer doesn't wait for it). It runs on the GPU by default; `create_nexto_bot(path, {.device = torch::kCPU})` runs it on the CPU. Its options also set how random it plays (`beta`: `1`, the default, picks the best action, and `0.5` samples from its policy) and its kickoff behavior; see `NextoOptions` in `src/external/nexto/nexto_bot.hpp`.

Nexto also plays in evaluation (see [Evaluation](evaluation.md)); training against it isn't supported. Other bots can be added through the interface in `src/external/external_bot.hpp`. Nexto is included under the MIT License with permission from its author; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
