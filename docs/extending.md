# Extending the Framework

**Rewards:** Subclass `reward::RewardFunction` and implement `get_reward(agent, current_state, previous_state)` and `clone()`. Optionally override `reset(initial_state)`, and `pre_tick(agents, current_state, previous_state)`, which runs once per tick before `get_reward` is called for any agent (useful for computing state shared across agents). Rewards are called once per agent per physics tick. Each env holds one instance shared by all of its agents, so store any per-agent state indexed by `agent.car_id`. Add a factory function to `src/reward/rewards.hpp`.

**Team Rewards:** For rewards that only depend on the team (goals, possession, ball toward goal), subclass `reward::TeamRewardFunction` and implement `get_team_reward(team, current_state, previous_state)` instead of `get_reward`. It is called once per team per tick, and every agent on that team receives the value. Team rewards are detected automatically. `self_share` has no effect on them, since teammates already receive the same value. With `opp_scale`, the opponent team's value is subtracted (not its total over its players), so `GoalReward` returning 1 to the scoring team with `opp_scale = 1` gives +1 and −1 in every match layout.

**Metrics:** Subclass `metric::MetricFunction` and implement `get_metric(agent, current_state, previous_state)` and `clone()`. Metrics are called once per agent per physics tick and return a `std::optional<double>`. Each value is added to a running sum and count, and the logged result is the sum divided by the count, so a metric returning 1 or 0 logs the fraction of ticks where its condition held. Returning `std::nullopt` leaves that tick out of both the sum and the count, which is how a metric can average only over the ticks it applies to (for example, ball speed only on ticks where the agent touched it). `get_metric` is `const`, so metrics can't keep state between ticks. Use `previous_state` for anything that compares against the last tick. Add a factory function to `src/metric/metrics.hpp`.

```cpp
class InAirRatioMetric : public metric::MetricFunction {
public:
    std::optional<double> get_metric(
        const env::AgentMeta& agent,
        const ffi::ArenaState& current_state,
        const ffi::ArenaState& /* previous_state */
    ) const override {
        // 1 when airborne, 0 on the ground. The logged average is the fraction of time spent in the air.
        return current_state.cars[agent.car_id].is_on_ground ? 0.0 : 1.0;
    }

    [[nodiscard]] std::unique_ptr<metric::MetricFunction> clone() const override {
        return std::make_unique<InAirRatioMetric>(*this);
    }
};
```

Only the agents being trained are measured. In PFSP envs, the past version's cars are left out. A metric that returns the same value for every agent, such as the ball's height, is still counted once per measured agent, so envs with more learner agents (for example 3v3 compared to 1v1) carry more weight in its average. Metrics are averaged per tick, so per-episode statistics (such as average episode length) aren't supported.

**State Setters:** Subclass `state::StateSetter` and implement `apply(arena, current_state, agents, rng)` and `clone()`. Add a factory function to `src/state/setters.hpp`.

New `.cpp` files must be added to the explicit source list in `CMakeLists.txt`.

**Observations & Actions:** The obs size and action space size are read from the builders, so the network adapts to changes automatically. The RLBot bot uses the obs and action code it was built with, so deploy a model with the code it was trained with (see [Deploying to RLBot](deploying.md)).

## Arena State Reference

Rewards, metrics, and state setters all read the game through `ffi::ArenaState` (declared in `include/buta_ppo/ffi.h`). It holds the world state after the current tick. It is never mirrored, so blue's goal is always at negative Y no matter which team an agent is on. Distances are in Unreal units (uu), velocities in uu/s, and angular velocities in rad/s. Index cars with `agent.car_id`.

| Field | Description |
| --- | --- |
| `ball.phys` | The ball's physics state (see `PhysState` below). |
| `cars[i].phys` | Car `i`'s physics state. |
| `cars[i].team` | `0` for blue, `1` for orange. |
| `cars[i].boost` | Boost amount, `0` to `100`. |
| `cars[i].is_on_ground`, `has_jumped`, `has_double_jumped`, `has_flipped`, `has_flip_or_jump`, `is_jumping`, `is_flipping`, `is_auto_flipping`, `is_supersonic` | Car movement flags. `has_flip_or_jump` is true while the car can still jump or flip. |
| `cars[i].air_time_since_jump`, `handbrake_val` | Seconds airborne since the last jump, and the handbrake amount. |
| `cars[i].is_demoed`, `demo_respawn_timer` | Whether the car is demolished, and the seconds until it respawns. |
| `boost_pads.big[6]`, `boost_pads.small[28]` | Seconds until each pad is available again. `0` means it can be picked up now. Pads are sorted by Y, then X. |
| `num_cars`, `tick_count` | Number of cars in the arena, and the arena's tick counter. |

**`PhysState`:** `pos[3]`, `vel[3]`, `ang_vel[3]`, and `rot_mat[3][3]`, where `rot_mat[0]` is the forward direction, `rot_mat[1]` is right, and `rot_mat[2]` is up. To work in angles, `util::math::euler_to_mat3` and `mat3_to_euler` (`src/util/math_utils.hpp`) convert between `rot_mat` and pitch, yaw and roll in radians, in the game's convention, the same as RLBot's `Rotator` and RLGym: positive pitch raises the nose, positive yaw turns toward the car's right (from +X toward +Y), and positive roll lowers its right side.

**Events (`events`):** Describe what happened during the current tick only.

| Field | Description |
| --- | --- |
| `is_ball_scored` | A goal was scored this tick. Check which goal from the ball's Y position. |
| `car_hit_ball[i]` | Car `i` touched the ball. |
| `ball_hit_extra_vel[i]` | Size of the extra velocity car `i`'s touch added to the ball. |
| `car_got_boost[i]` | Car `i` picked up a boost pad. |
| `car_bumped_car[b][v].bumped` | Car `b` bumped car `v`. Only real bumps count (the bumper drives into the victim, inside its bump cone), not every contact between cars. A head-on collision can register in both directions. |
| `car_bumped_car[b][v].is_demo` | The bump demolished car `v`. Only meaningful when `.bumped` is true. |

For example, a reward for bumping opponents:

```cpp
float get_reward(const env::AgentMeta& agent, const ffi::ArenaState& state, const ffi::ArenaState& /*prev_state*/) override {
    float reward = 0.0f;
    for (uint32_t v = 0; v < state.num_cars; v++) {
        const ffi::CarBump& bump = state.events.car_bumped_car[agent.car_id][v];
        if (bump.bumped && state.cars[v].team != state.cars[agent.car_id].team) {
            reward += bump.is_demo ? 1.0f : 0.5f;
        }
    }
    return reward;
}
```

## Directory Structure

```text
buta-ppo/
├── assets/
│   └── collision_meshes/    # Required RocketSim arena geometry (not tracked)
├── bot/                     # RLBot bot directory, filled by deploy.py (only the .toml templates are tracked)
├── checkpoints/             # Saved models, optimizer state, and the PFSP pool
├── docs/                    # Documentation
├── include/buta_ppo/        # FFI declarations shared with the Rust bridge
├── logs/                    # TensorBoard event files
├── rust_bridge/             # Rust FFI wrapper for RocketSim and the RLBot bot binary
├── src/
│   ├── action/              # Discrete action table and action masks
│   ├── env/                 # Single environment and the vectorized env pool
│   ├── external/            # Bots from outside the framework (Nexto) that drive a team in the visualizer and in evaluation
│   ├── metric/              # Metric functions and the metric manager
│   ├── obs/                 # Observation builder and ball prediction
│   ├── reward/              # Reward functions and the reward manager
│   ├── rl/                  # Network, rollout buffer, trainer, optimizers, PFSP, evaluation, and the training runner
│   ├── state/               # State setters for episode resets
│   ├── util/                # Inline math and vector operations
│   ├── bot_api.cpp          # Inference-only library used by the RLBot bot
│   └── main.cpp             # Entry point and all training configuration
├── deploy.py                # Builds and packages the RLBot bot
└── THIRD_PARTY_NOTICES.md   # Notices for included third-party material (Nexto)
```
