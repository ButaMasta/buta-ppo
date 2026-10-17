### Current Unreleased Changes
1. Added metrics logging via a custom system. The user can implement an abstract class to create their own metrics.
    - Non-breaking. No resolutions need to be made.
2. Added `steps_per_log` to `RunnerConfig` to set how many lifetime steps pass between TensorBoard logs. Each log aggregates every update since the previous one, and a partial window is logged when training stops. The default of `0` logs every update, as before. Console output is unchanged.
    - Non-breaking. Existing configs keep logging every update.
    - The per-slot `PFSP/Slot_<n>_Version` and `PFSP/Slot_<n>_Win_Rate` tags are no longer logged, since a slot plays a different snapshot each update. `PFSP/Win_Rate`, `PFSP/Games`, and `PFSP/Pool_Size` remain, and the console still lists each update's opponents.
    - `Performance/Rollout_SPS` no longer includes time spent writing logs, so it reads slightly higher than before.
3. Added an optional `pre_tick(agents, current_state, previous_state)` method to `RewardFunction`. It is called once per physics tick before `get_reward` is called for any agent, so rewards can compute state shared across agents for that tick.
    - Non-breaking. The default implementation does nothing, so existing rewards need no changes.

#### 0.1.1
1. Updated the logging with a patch via cmake to replace the dependency's invalid `int32_t` logging for the x-axis to be `int64_t`. 
    - Non-breaking change but existing `.tfevents` files with already faulty negative values cannot be retroactively updated.
    - Resolutions:
        - Deleting the file is an accepatable resolution.
        - Keeping it will just mean the new values get added as expected.

#### 0.1.0
Initial release.