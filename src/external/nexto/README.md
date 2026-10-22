# Nexto

A port of Nexto, the RLGym community bot by Rolv-Arild and the Necto contributors
(https://github.com/Rolv-Arild/Necto), as it plays in the RLBot botpack. Included under the MIT License with
permission; see [THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md).

| File | What it is |
|---|---|
| `nexto-model.pt` | The botpack Nexto's TorchScript model, unmodified. SHA-256 `bf5343b5eeacac6bf7cdb75dac4a5c14ba0f94d820eae75f00a211b6119d69fa` |
| `nexto_bot.hpp`, `nexto_bot.cpp` | C++ translation of the botpack's `bot.py`, `agent.py` and `nexto_obs.py` |
| `nexto_net.hpp`, `nexto_net.cpp` | The model's network rebuilt natively (runs on CUDA), with its weights loaded from `nexto-model.pt` |

Use it with `create_nexto_bot("src/external/nexto/nexto-model.pt")` from `src/external/bots.hpp` (paths are relative
to the repo root, where buta-ppo runs).
