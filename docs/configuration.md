# Configuration

All configuration lives in `src/main.cpp`, in the `RunnerConfig` struct. The values there are demo values that show the available options, not tuned hyperparameters.

## Matches, State Setters, Rewards & Metrics

```cpp
// Match types, each with a weight for its share of the envs.
config.match_distributions = {
    {1, 1, 0.5f},   // 1v1 at 50% weight.
    {2, 2, 0.5f}    // 2v2 at 50% weight.
};
// The obs has a fixed number of car slots, so this must fit the largest team.
config.max_players_per_team = 2;

// How episodes start, each with a weight.
config.setter_distributions = {
    {create_default_kickoff_setter(), 0.30f},               // Regular kickoffs at 30%.
    {create_random_state_setter(true, true, true), 0.35f},  // Random, with cars on the ground, at 35%.
    {create_random_state_setter(true, true, false), 0.35f}  // Fully random at 35%.
};

// Besides a goal, episodes end after this long without a touch, or once they last this long. 0 turns a limit off.
config.no_touch_timeout_seconds = 10.0f;
config.episode_time_limit_seconds = 150.0f;

// Rewards with their names (for logging) and weights. Rewards are computed every physics tick.
config.reward_entries = {
    {"VelocityToBall", create_velocity_to_ball_reward(), 0.1f},
    {"TouchBall", create_touch_ball_reward(), 1.0f},
    {"Goal", create_goal_reward(), 10.0f, {.opp_scale = 1.0f}}  // Zero-sum, see Reward Options below.
};

// Optional metrics with their names (for logging). Metrics are computed every physics tick but don't affect training.
config.metric_entries = {
    {"InAirRatio", create_in_air_ratio_metric()}
};
```

**Episode Limits:** Besides a goal, an episode ends when no car has touched the ball for `no_touch_timeout_seconds`, or once it has lasted `episode_time_limit_seconds`. These are truncations, not terminations: the critic's estimate of the last state stands in for the rest of the episode, so an episode that is cut short isn't treated as worth nothing from there on. A short no-touch timeout keeps slow, ball-less play out of the training data, but the bot then rarely learns from it. `Episode/Truncations_No_Touch` and `Episode/Truncations_Time_Limit` in [Telemetry](telemetry.md) show how often each limit ends episodes. `0` turns a limit off, a negative value stops training at startup, and a no-touch timeout longer than the time limit prints a `[WARN]`, since it can never fire.

Rewards are summed over the `ticks_per_step` ticks of each step, so per-tick rewards scale with `ticks_per_step`.

**Reward Options:** Each entry can take an optional `RewardOptions` that mixes its values between agents. They are applied to each step's weighted values, after the reward is computed for every agent.

```cpp
{"TouchBall", create_touch_ball_reward(), 1.0f, {.self_share = 0.7f, .opp_scale = 1.0f}}
```

- `self_share` (default `1`, in `[0, 1]`): the fraction of an agent's own reward that it keeps. The rest is split evenly among its teammates. An agent on team *T* with *n* players receives `self_share · r + (1 − self_share) · (mean of its teammates' r)`. The team's total is unchanged, and the agent keeps the same fraction of its own reward in every match layout, but each teammate's share shrinks as teams get bigger. Agents without teammates keep their whole reward.
- `opp_scale` (default `0`, `≥ 0`): subtracts the opponent team's total, split across the agent's team: `r − opp_scale · S_opp / n`. At `1`, the two teams' totals are exactly opposite (zero-sum), including with uneven teams. A team with no opponents (for example 1v0) is unaffected.

| `self_share = 0.5` | Agent keeps | Each teammate gets |
|---|---|---|
| 1v1 | all of it | — |
| 2v2 | half | half |
| 3v3 | half | a quarter |

The order of the two options doesn't matter. Don't set `opp_scale` on a reward that already punishes the other team (returning −1 to the team that conceded, for example), since that doubles it. `Reward_Magnitude/` in [Telemetry](telemetry.md) shows when a reward is larger than expected.

Invalid options, non-finite weights, and duplicate reward names stop training at startup. Settings that have no effect, such as `self_share` in 1v1-only layouts, a zero weight, or an `opp_scale` above 1, print a `[WARN]` instead.

Metrics are averaged over every tick since the previous TensorBoard log, so they don't depend on `ticks_per_step`. Leaving `metric_entries` empty disables them. A metric can also count what a reward entry judges each tick by naming it (see "Metrics that read rewards" in [Extending the Framework](extending.md)). To write your own metrics see [Extending the Framework](extending.md) and [Telemetry](telemetry.md) for how they are logged.

**Metric Options:** Every metric runs once per measured agent per physics tick, and each costs a few nanoseconds per call, so a long list of metrics slows the env workers measurably. Each entry can take an optional `MetricOptions` that limits where it is computed, and `metric_env_share` measures every entry in a share of the envs.

```cpp
// Compute the metrics in a quarter of each match layout's envs.
config.metric_env_share = 0.25f;

config.metric_entries = {
    {"InAirRatio", create_in_air_ratio_metric()},
    // Only in 3v3 envs. The other envs never call it.
    {"InAirRatio_3v3", create_in_air_ratio_metric(), {.layouts = {{3, 3}}}},
    // A {blue, orange} pair matches either way round, so {1, 2} covers 1v2 and 2v1.
    {"InAirRatio_1v2", create_in_air_ratio_metric(), {.layouts = {{1, 2}}}},
    // Measured in every env whatever metric_env_share is, for a metric of rare events that needs every sample.
    {"InAirRatio_AllEnvs", create_in_air_ratio_metric(), {.env_share = 1.0f}}
};
```

- `layouts` (default empty, every layout): the team sizes `{blue, orange}` of the match layouts the entry is computed in. Envs of other layouts skip it entirely, which saves its cost there and makes any metric loggable per layout. An entry whose layouts no match distribution plays prints a `[WARN]`, since it never logs.
- `env_share` (default unset, `metric_env_share`): the share of each layout's envs, in `(0, 1]`, that compute the entry. `metric_env_share` (default `1`) is the share for entries that don't set their own.

Shares pick `max(1, round(share × n))` of a layout's `n` envs, spread evenly by env index, counted separately for its self-play and its PFSP envs so each kind gets its share. The picks are fixed for the run, so a restart measures the same envs. A metric's logged value is then the average over the measured envs' agent ticks: the same quantity, from fewer samples, so a share of 0.25 cuts the metrics' cost about four times and their curves get somewhat noisier, mostly for rare events. Shares outside `(0, 1]` and layouts with an empty team stop training at startup. With any option in use, startup prints how many envs compute metrics and how many entries each computes.

## Environments & Batch Size

```cpp
config.num_threads = 0;             // Worker threads stepping the envs. 0 = one per CPU, minus one.
config.num_envs = 0;                // Total envs. 0 = picked from target_steps_per_agent. Training needs at least 2.
config.ticks_per_step = 8;          // Physics ticks (120 per second) each action is held for.

config.target_steps_per_update = 50'000;
config.num_minibatches = 1;
config.target_steps_per_agent = 100; // Steps each agent collects per update, which sets the env count.
```

With `num_envs = 0`, the env count is the one whose steps per agent (`target_steps_per_update` divided by the learner agents) come closest to `target_steps_per_agent`. More envs are usually faster, but each update then learns from shorter stretches of play per agent. Setting `num_envs` above 0 uses that count instead. Workers take envs one at a time as they finish, so `num_envs` doesn't need to be a multiple of `num_threads`, and `num_threads = 0` uses one worker per CPU in `worker_cpus`. When that's empty, it uses one per CPU the process may run on except one, which leaves the main thread a CPU of its own.

The actual steps per update are rounded up from `target_steps_per_update`, so every agent contributes the same number of steps and the total divides evenly into `num_minibatches`. The update size is limited by VRAM. Start with a small `target_steps_per_update` or more minibatches, and increase from there.

To find the fastest thread count, CPU pinning, and env count for your machine, see [Performance](performance.md).

## Network

The actor-critic network is a shared MLP trunk that branches into separate actor and critic MLPs. Any of the three can be empty. For example, `shared_layers = {}` gives fully separate actor and critic networks.

```cpp
config.ac_cfg.shared_layers = { 1280, 1280, 1024 };
config.ac_cfg.actor_layers  = { 768, 512 };
config.ac_cfg.critic_layers = { 1024, 768 };
config.ac_cfg.use_layer_norm = true;
```

`use_layer_norm` (a LayerNorm between each hidden layer and its ReLU) defaults to `true`.

## PPO & Optimizer

```cpp
config.ppo_cfg.epochs = 1;
config.ppo_cfg.entropy_coef = 0.035f;
// Also available: clip_ratio, value_coef, max_grad_norm, gae_gamma, gae_lambda.

// Presets: OptimizerConfig::adam(lr), ::adamw(lr, wd), ::muon(muon_lr, adamw_lr, wd).
config.ppo_cfg.optimizer = OptimizerConfig::muon(3e-4, 3e-4, 0.01);
// Sections (shared / actor / critic) can be adjusted afterwards:
config.ppo_cfg.optimizer.critic.with_lr(1e-4f);
```

The Muon preset applies Muon to hidden-to-hidden weight matrices and AdamW to everything else. Its learning rate is scaled so values comparable to Adam's carry over.

### Learning Rate Schedule

`config.ppo_cfg.lr_schedule` multiplies every learning rate in `optimizer` by a factor that changes with lifetime steps, so the ratios between sections and groups stay as set. It holds its start scale until its start step, moves to its end scale by its end step, and holds that after. The default is 1 throughout.

```cpp
// Halve twice between 7B and 12B steps, at an even pace: 0.5 at 9.5B, 0.25 from 12B on.
config.ppo_cfg.lr_schedule = LrSchedule::exponential(7'000'000'000, 12'000'000'000, 1.0f, 0.25f);
// Also LrSchedule::linear and LrSchedule::cosine, with the same arguments.
```

The scale is set before each update and read by the captured training graph, so it changes smoothly without restarts, and a resumed run picks it up from its step count. A schedule fits a [training plan](training.md#training-plans) stage: set it in the stage it should run in. The scale is logged as `Policy/LR_Scale`, and printed under each update while a schedule is set.
