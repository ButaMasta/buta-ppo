# Telemetry

Metrics are written to `logs/<bot_name>.tfevents`, with lifetime steps on the x-axis.

```cpp
// Lifetime steps between TensorBoard logs, each aggregating the updates since the last. 0 logs every update.
config.steps_per_log = 0;
```

By default, every update is logged. With a larger `steps_per_log`, each logged point covers every update since the previous one: rewards and metrics are averaged over all of its steps and ticks, episode and PFSP counts are totals, the training statistics are averaged over its updates, and steps per second covers its whole duration. Log points land on multiples of `steps_per_log`, so they stay on schedule across resumes. A value below the actual steps per update logs every update. When training stops, any updates logged since the last point are written as one final, shorter point. The console output is still printed every update.

| Group | Metrics |
|---|---|
| `Reward_Components/` | Each reward's average weighted value per step, before the reward options. Shows how often the rewarded behavior happens |
| `Reward_Magnitude/` | Each reward's average absolute value per step, after the reward options. Shows how much each reward drives training |
| `Reward_Share/` | Each reward's fraction of the total magnitude, summing to 1. Shows which rewards dominate training. Not logged when no reward was received |
| `Reward/` | `Mean_Step` and `Mean_Abs_Step`, the totals of `Reward_Components/` and `Reward_Magnitude/` |
| `Episode/` | Goals, and truncations split into no-touch and time-limit, summed since the previous log |
| `Metrics/` | Each configured metric's average over every measured tick since the previous log. A metric that skipped every tick in that time isn't logged for it |
| `Loss/` | Policy, value, entropy |
| `Policy/` | Approx KL, clip fraction, normalized entropy |
| `Value/` | Explained variance (how much of the returns the critic predicts, independent of reward scale) |
| `PFSP/` | Win rate and games since the previous log, and pool size. The versions played each update are printed to the console |
| `Skill/` | Logged at each evaluation, per team size: `<N>vN_Elo` and its standard error `<N>vN_Elo_Std_Error`, `<N>vN_Score` (wins plus half the draws), `<N>vN_Draw_Rate`, and `<N>vN_Mean_Game_Seconds`, over the games against the snapshot anchors (none on a run's first evaluation with Nexto, which plays only Nexto); with Nexto, `<N>vN_Nexto_Beta`, the rung played (where the placement search ended on the first evaluation), `<N>vN_Nexto_Score`, the score against it, and `<N>vN_Nexto_Gap` and `<N>vN_Nexto_Gap_Std_Error` (see [Evaluation](evaluation.md)); plus `Active_Anchors`, the anchors played. The console prints the same after the update it ran in, and the Nexto rungs played with their results |
| `Performance/` | Total and rollout steps per second, counted in learner agent steps and not including the time spent writing logs and checkpoints, and `Rollout_Physics_Wait`, the fraction of the rollout spent waiting on physics (see [Finding the Rollout Bottleneck](performance.md#finding-the-rollout-bottleneck)). `Env_Step_Microseconds` is the worker time per env step, and `Touch_Ticks_Per_Env_Step` and `Repredictions_Per_Env_Step` count the ball touches and full ball predictions that drive it, so a drop in SPS can be told apart from a bot that plays the ball more. `Eval_Time` is the length of each evaluation, which isn't counted in the steps per second |

Rewards are averaged over learner agent steps, so PFSP opponents are left out. A reward that itself returns opposite values to the two teams (for example −1 to the team that conceded) averages to about 0 in `Reward_Components/` during self-play, since both teams are learners. Making it zero-sum with `opp_scale` instead keeps `Reward_Components/` meaningful, since it is logged before the options. `Reward_Magnitude/` doesn't cancel either way. The absolute value is taken per step, matching what PPO trains on, so a reward that flips sign within one step counts as their sum.

Approx KL and clip fraction are measured during each minibatch's forward pass, before its update. With `epochs = 1` and `num_minibatches = 1`, they only reflect small precision differences, not how far the update moved the policy.

Normalized entropy is each sample's entropy divided by its maximum, `ln(valid actions)`, averaged over the rollout. It runs from 0 (always the same action) to 1 (uniformly random over the actions the mask allows). The raw entropy also moves when the mix of masks changes, for example with more time in the air (about twice as many valid actions as on the ground) or more time without boost. Normalized entropy only moves when the policy itself becomes more or less random.

To view the logs with TensorBoard:

1. **Create a Virtual Environment** anywhere you like, and activate it. Activation has to be repeated in every new terminal, and the `activate` file depends on your shell.
```bash
python -m venv .venv
source .venv/bin/activate
```
2. **Install TensorBoard:**
```bash
pip install tensorboard
```
3. **Run TensorBoard:**
```bash
tensorboard --logdir=/path/to/logs --host 0.0.0.0 --port=6006
```
