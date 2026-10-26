# PFSP

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
