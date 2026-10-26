# Visualizer Mode

Setting `config.render = true;` in `main.cpp` skips training and runs a single environment in real time in the RocketSim visualizer, using the latest checkpoint for `bot_name`. By default the policy plays its most likely action, as the RLBot bot does, so the visualizer shows how the bot plays in a match. `config.render_best_action = false;` samples actions from the policy like training does instead.

## Playing Against Nexto

The visualizer can put Nexto, the RLGym community bot, on a team in place of your policy:

```cpp
auto nexto = create_nexto_bot("src/external/nexto/nexto-model.pt");
config.render_external_bots = {{nexto, ffi::Team::Orange}};  // Your bot plays blue.
```

List both teams for Nexto against itself. Nexto plays as the RLBot botpack version does, with its own observations, actions, decision timing, and scripted kickoff, and it plays through the 3 second countdown before each kickoff as it would in game (the visualizer doesn't wait for it). It runs on the GPU by default; `create_nexto_bot(path, {.device = torch::kCPU})` runs it on the CPU. Its options also set how random it plays (`beta`: `1`, the default, picks the best action, and `0.5` samples from its policy) and its kickoff behavior; see `NextoOptions` in `src/external/nexto/nexto_bot.hpp`.

Nexto also plays in evaluation (see [Evaluation](evaluation.md)); training against it isn't supported. Other bots can be added through the interface in `src/external/external_bot.hpp`. Nexto is included under the MIT License with permission from its author; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
