# Evaluation

Evaluation tracks how good the bot is over training, which rewards and losses can't show. Every `steps_per_eval` steps, training pauses for a few seconds while the learner plays fixed past versions of itself (the anchors) and Nexto, the RLGym community bot, and it gets an Elo rating per team size in TensorBoard. The ratings are on Nexto's scale: Nexto playing its best action is 1500 Elo in every team size, so a rating means the same in every run and for every bot built with the framework.

```cpp
config.eval_cfg.steps_per_eval = 50'000'000;   // 0 turns evaluation off.
```

**The games.** Each start state in a fixed bank of kickoffs is played once from each side against each anchor and against Nexto, from the kickoff to the first goal. Every car in a kickoff is moved by up to 2.5 uu and turned by up to 0.01 rad, so the bank holds many different games instead of the 5 standard kickoffs. Both sides play their most likely action, as the RLBot bot does. A game with no goal by `game_cap_seconds` is a draw. The same bank is played every time, and the same models always give the same results, so a change in rating comes from the learner and not from luck.

**The ratings.** The learner's results are fitted with the Bradley–Terry model, in which the chance of beating a player depends only on the difference in rating, shown on the Elo scale: 400 points is 10:1 odds, 200 points about 3:1. Draws count as half a win. Nexto at its best is the fixed reference, 1500 Elo in every team size it plays (1v1 to 3v3). Each team size is rated on its own scale, so a 1v1 rating can't be compared with a 3v3 one: 1500 means Nexto's level in that mode. With `nexto = false`, and in team sizes Nexto doesn't play (4v4), the first evaluation instead makes the current learner the first anchor, the reference at 0 Elo, so ratings measure progress since evaluation started.

**Nexto and the ladder.** Most bots are far weaker than Nexto for a long time, and games a bot loses every time say nothing about how far behind it is. So evaluation also has weaker versions of Nexto, the ladder: Nexto playing more randomly, set by its `beta` (1 plays its best action, 0.5 samples from its policy, 0 picks uniformly at random), at 13 rungs from 1 down to 0. Each rung's rating was measured once from Nexto-vs-Nexto games against the rungs above it, and ships with the framework in `src/external/nexto/nexto_ladder.txt`:

| `beta` | 1v1 | 2v2 | 3v3 |
|---|---|---|---|
| 1 | 1500 | 1500 | 1500 |
| 0.75 | 1457 ± 11 | 1397 ± 11 | 1307 ± 13 |
| 0.5 | 1422 ± 10 | 1294 ± 11 | 1160 ± 13 |
| 0.375 | 1369 ± 13 | 1177 ± 14 | 1034 ± 16 |
| 0.25 | 1247 ± 14 | 987 ± 16 | 774 ± 18 |
| 0.2 | 1190 ± 16 | 860 ± 18 | 585 ± 21 |
| 0.15 | 1042 ± 17 | 668 ± 20 | 338 ± 23 |
| 0.125 | 926 ± 19 | 524 ± 21 | 164 ± 25 |
| 0.1 | 791 ± 20 | 337 ± 23 | -47 ± 26 |
| 0.075 | 561 ± 23 | 94 ± 25 | -320 ± 29 |
| 0.05 | 206 ± 27 | -260 ± 29 | -648 ± 31 |
| 0.025 | -224 ± 32 | -582 ± 32 | -930 ± 34 |
| 0 | -286 ± 33 | -648 ± 32 | -990 ± 34 |

Each evaluation, the learner plays one rung per team size, the one closest to its own rating, which says the most about it. The first evaluation of a run finds that rung with a short search over the ladder, a few rounds of Nexto games, so it takes longer than the rest. Nexto plays as in the RLBot botpack, with its own decision timing, its scripted kickoff, the countdown before the kickoff, and each car joining the match at its own moment; its games repeat exactly like the anchors'. Nexto's games are as fast as the anchors', since many of them are played at once: about 1.5 to 3 s per evaluation, longer against weak rungs, whose games last longer.

**The anchors.** The learner becomes an anchor once its score (draws count half) against the newest anchor, over all team sizes, reaches `promote_win_rate`. An anchor's rating is the one the learner had when it was promoted, and never changes, so earlier points on the curve never move. Games against an anchor the learner is far ahead of say little, so an anchor isn't played while the learner's predicted score against it, from their ratings, is above `retire_win_rate`. It's played again if the learner falls back. The newest anchor is always played.

| Option | Default | Description |
|---|---|---|
| `steps_per_eval` | `50'000'000` | Lifetime steps between evaluations. `0` turns evaluation off. |
| `team_sizes` | `{}` | Team sizes rated, each played as NvN. Empty rates every even layout in `match_distributions`; with none, startup fails until you set the sizes or turn evaluation off. |
| `bank_size` | `64` | Start states per team size. Each is played from both sides against every anchor played. |
| `bank_seed` | `12345` | Seed of the start-state bank. |
| `pos_jitter` | `2.5` | How far each kickoff car is moved, in uu. |
| `yaw_jitter` | `0.01` | How far each kickoff car is turned, in radians. |
| `game_cap_seconds` | `150` | A game without a goal by then is a draw. |
| `active_anchors` | `4` | Most anchors played at once. When more qualify, the newest are played. |
| `promote_win_rate` | `0.7` | The learner's score against the newest anchor that makes it an anchor. |
| `retire_win_rate` | `0.85` | Anchors the learner is predicted to score above this against aren't played. Must be above `promote_win_rate`. `1` plays every anchor, up to `active_anchors`. |
| `virtual_draws` | `1` | Draws added against every anchor when fitting, which keeps a rating finite after a perfect record. At least 1. |
| `nexto` | `true` | Rate on Nexto's scale, playing a Nexto rung per team size each evaluation. `false` rates against the first anchor at 0. |
| `nexto_model_path` | `"src/external/nexto/nexto-model.pt"` | Nexto's model. |
| `nexto_ladder_path` | `"src/external/nexto/nexto_ladder.txt"` | The ladder's ratings, measured for the default `pos_jitter`, `yaw_jitter`, and `game_cap_seconds`. |
| `nexto_calibration_bank_size` | `512` | Kickoffs per team size and pairing of rungs in a precise calibration (`calibrate_nexto_ladder`, below). Only read while one runs. |
| `nexto_calibration_rungs_above` | `2` | How many rungs above each rung a precise calibration plays it against. Only read while one runs. |

Changing the bank, the jitter, `game_cap_seconds`, `virtual_draws`, `ticks_per_step`, `max_players_per_team`, the network shape, or `nexto` makes old ratings incomparable, so evaluation starts over (see [Checkpoints & the PFSP Pool](training.md#checkpoints--the-pfsp-pool)). The other options only change which games are played and can be changed at any time.

The ladder's ratings depend on the kickoff jitter and `game_cap_seconds`. With other values than the defaults, the ladder is measured again once at startup, which takes about 90 s and is announced on the console, and saved as `checkpoints/nexto_ladder.txt` for every later run with the same values. That measurement uses fewer games than the shipped one, so its ratings are less precise, about ±30 Elo near Nexto and up to ±130 at the bottom. The console notes this at startup, with that ladder's numbers, since the ratings' error bars don't include it.

For a precise ladder with your own values, set `RunnerConfig::calibrate_nexto_ladder = true`, run once instead of training (about 20 minutes with the default `nexto_calibration_bank_size` and `nexto_calibration_rungs_above`), and set it back. It calibrates for the jitter and `game_cap_seconds` in `eval_cfg` and writes `checkpoints/nexto_ladder.txt`, where evaluation looks for a ladder for non-default values, so the shipped file is never overwritten (with a changed `nexto_ladder_path`, it writes that file instead). Do it before training with those values: a new ladder changes the ratings' scale, so an evaluation ledger rated with another ladder is moved aside and the Elo curve starts over. If you clear `checkpoints/` often, for example to try different models with the same evaluation settings, keep the ladder outside it: set `eval_cfg.nexto_ladder_path` to a file of your own, such as `"nexto_ladder_custom.txt"`, for both the calibration run and training. The calibration writes that file, evaluation reads it first, and every run rated against it shares one scale, so their ratings compare. A file calibrated for other values than the current ones is ignored, not overwritten, so each combination of jitter and game cap needs its own. With the default values, `calibrate_nexto_ladder` regenerates the shipped file instead, which is only needed after changing Nexto's model, the rungs, or the kickoff setter.

**Reading the curve.** Each point has an error bar, `Skill/<N>vN_Elo_Std_Error`: about ±20 to 25 Elo with one anchor and a Nexto rung played, for the default bank. It doesn't include the error of the rung's own rating (the `±` in the table), which shifts every rating made against that rung alike, so ratings against the same rung compare more precisely than their distance to Nexto. Differences smaller than about two error bars are noise. Swings much larger than that are real changes in how the bot plays. Early in training most games are draws, since neither bot can score, and draws count half, so early ratings stay close together even when the bot is clearly improving.

**The Nexto gap.** `Skill/<N>vN_Nexto_Gap` is the learner's rating from its Nexto games alone minus its rating from its snapshot games alone, in Elo, with its own error bar, `<N>vN_Nexto_Gap_Std_Error`. It checks that the two kinds of opponents agree on the learner's level, which the Elo curve, fitted to both at once, can't show. Near zero, within about two error bars, they agree. Persistently negative means the learner does worse against Nexto than its snapshots say, for example by beating its past versions through habits they share, so its distance to Nexto is overstated; persistently positive means the opposite, for example by exploiting the weak rungs' random play. Look at the trend over many evaluations, not single points. It's logged once both were played, from a run's second evaluation on, and only while the learner's score against the rung is between 0.05 and 0.95, about 510 Elo either side of it. Past the ladder's ends, far above Nexto at beta 1 or far below beta 0, a near-perfect record can't say how far past the rung the learner is, so the gap would grow without meaning anything.

**The cost.** On an i5-14600K with an RTX 5070 and three team sizes, an evaluation took about 4.5 s with one anchor and a Nexto rung played, and the first one of a run, which places the learner on the ladder, about 11 s. Each extra anchor adds about 1.3 s. That's about 4% of training time at the default interval. `Performance/Eval_Time` logs it. The time grows with the anchors played, the team sizes, and `bank_size`, and each phase (the anchors' games, then Nexto's) lasts until its slowest game ends. The evaluation games run on their own worker threads, on `worker_cpus` like training. Evaluation keeps every game it can play built for the whole run, about 2 MiB of RAM each: about 3.6 GB with the defaults, plus 1.2 GB for Nexto's games, and Nexto's network and CUDA graphs take about 550 MiB of GPU memory.
