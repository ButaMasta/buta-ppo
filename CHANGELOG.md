### Current Unreleased Changes
1. Added metrics logging via a custom system. The user can implement an abstract class to create their own metrics.
    - Non-breaking. No resolutions need to be made.
2. Added `steps_per_log` to `RunnerConfig` to set how many lifetime steps pass between TensorBoard logs. Each log aggregates every update since the previous one, and a partial window is logged when training stops. The default of `0` logs every update, as before. Console output is unchanged.
    - Non-breaking. Existing configs keep logging every update.
    - The per-slot `PFSP/Slot_<n>_Version` and `PFSP/Slot_<n>_Win_Rate` tags are no longer logged, since a slot plays a different snapshot each update. `PFSP/Win_Rate`, `PFSP/Games`, and `PFSP/Pool_Size` remain, and the console still lists each update's opponents.
    - `Performance/Rollout_SPS` no longer includes time spent writing logs, so it reads slightly higher than before.
3. Added an optional `pre_tick(agents, current_state, previous_state)` method to `RewardFunction`. It is called once per physics tick before `get_reward` is called for any agent, so rewards can compute state shared across agents for that tick.
    - Non-breaking. The default implementation does nothing, so existing rewards need no changes.
4. Exposed RocketSim's car-to-car bump events. `ArenaState::events.car_bumped_car[bumper][victim]` has `bumped` set when a car bumps another car that tick, and `is_demo` set when that bump demolished the victim. Also added an "Arena State Reference" section to the README describing the fields available in `ArenaState`.
    - Non-breaking. The new field is added at the end of `rs_arena_events`, and both sides of the FFI were updated together.
5. Added zero-sum, team-sharing, and team rewards.
    - `RewardEntry` takes an optional `RewardOptions`, applied to each step's weighted values:
        - `self_share` (default `1`) is the fraction of an agent's own reward it keeps. The rest is split among its teammates, so the agent keeps the same fraction in every match layout.
        - `opp_scale` (default `0`) subtracts the opponent team's reward. At `1`, the reward is zero-sum between the teams, including uneven ones.
        - Example: `{"TouchBall", create_touch_ball_reward(), 1.0f, {.self_share = 0.7f, .opp_scale = 1.0f}}`.
    - Rewards that only depend on the team can subclass `TeamRewardFunction` and implement `get_team_reward(team, ...)`, which is called once per team per tick. `GoalReward` is now a team reward that returns 1 to the scoring team, and the demo config uses it with `opp_scale = 1`.
    - Reward entries are checked at startup. Invalid options, non-finite weights, and duplicate reward names throw. Settings that have no effect (such as `self_share` on a team reward or in 1v1-only layouts, or a weight of 0) and an `opp_scale` above 1 print a `[WARN]`.
    - New TensorBoard groups:
        - `Reward_Magnitude/<name>`: the average absolute value per step that agents received, after the options.
        - `Reward_Share/<name>`: each reward's fraction of the total magnitude.
        - `Reward/Mean_Abs_Step`: the total magnitude.
    - `Reward_Components/` and `Reward/Mean_Step` now log values before the options are applied.
    - Non-breaking. Existing rewards and configs work unchanged, except for configs with duplicate reward names, which now throw since their telemetry used to merge silently. Resolution: give each entry a unique name.
    - `GoalReward` no longer returns −1 on its own, so it only punishes the conceding team when paired with `opp_scale = 1`. Configs that use it without `opp_scale` lose the conceding penalty. Resolution: add `{.opp_scale = 1.0f}` to the entry.
    - `Reward_Components/Goal` now logs goals scored, where it used to read about 0 during self-play, so it rises when resuming a run logged before this change.

#### 0.1.1
1. Updated the logging with a patch via cmake to replace the dependency's invalid `int32_t` logging for the x-axis to be `int64_t`. 
    - Non-breaking change but existing `.tfevents` files with already faulty negative values cannot be retroactively updated.
    - Resolutions:
        - Deleting the file is an accepatable resolution.
        - Keeping it will just mean the new values get added as expected.

#### 0.1.0
Initial release.