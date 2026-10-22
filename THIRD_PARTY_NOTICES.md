# Third-Party Notices

buta-ppo is licensed under the MIT License (see [LICENSE](LICENSE)). It includes the third-party material below.

## Nexto (from Necto)

Source: https://github.com/Rolv-Arild/Necto, by Rolv-Arild and the Necto contributors.

Included in buta-ppo:
- `src/external/nexto/nexto-model.pt`: the Nexto model weights from the RLBot botpack release of Nexto, unmodified.
- `src/external/nexto/nexto_bot.hpp` and `src/external/nexto/nexto_bot.cpp`: a C++ translation of Nexto's
  `bot.py` (tick timing, scripted kickoff), `agent.py` (action table and action selection) and `nexto_obs.py`
  (observation builder).
- `src/external/nexto/nexto_net.hpp` and `src/external/nexto/nexto_net.cpp`: a C++ reimplementation of the network
  in `nexto-model.pt`, loading that file's weights: the EARL perceiver from Rolv-Arild's EARL-pytorch
  (https://github.com/Rolv-Arild/EARL-pytorch, `earl_pytorch/model/earl_perceiver.py`) and Necto's output head
  (`training/agent.py`: `Necto`, `ControlsPredictorDot`), as traced in the model file.

The original Necto project is licensed under CC BY-NC-SA 4.0. Rolv-Arild gave permission in October 2026 to
include the material above in buta-ppo under the MIT License. This permission applies to the listed material as
distributed in buta-ppo; it does not change the license of the original Necto repository.

```
MIT License

Copyright (c) 2026 Rolv-Arild and the Necto contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## rlgym-compat

`src/external/nexto/nexto_bot.cpp` follows the game state decoding of rlgym-compat 1.0.2
(https://rlgym.github.io), by Lucas Emery and Matthew Allen, licensed under the Apache License 2.0, which the
botpack Nexto runs on: the on-ground grace ticks, flip availability and boost scale.
