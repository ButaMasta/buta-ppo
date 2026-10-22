// buta-ppo/src/external/external_bot.hpp
#pragma once

#include "buta_ppo/ffi.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>

namespace buta_ppo::external {

/**
 * @brief What RLBot's game info says about the current tick, beyond the arena state.
 */
struct GameInfo {
    // RLBot's `is_kickoff_pause`: from "go" until the ball is first touched. False during the countdown.
    // Currently only needed for Nexto. This struct will be extended for future external bots.
    bool kickoff_pause = false;
};

/**
 * @brief Drives one car for an external bot, packet by packet, the way RLBot calls the bot.
 *
 * For every packet the bot would get (one per physics tick, and one per tick of the frozen countdown before a
 * kickoff), the env calls `observe` once, then, if the controller `wants_decision`, builds its obs, has the bot infer
 * (batched with the bot's other cars in the env) and passes the result to `set_decision`, then calls `tick` for the
 * car's controls. A controller lives as long as its env and isn't reset between episodes, like a bot that stays in
 * the match across goals.
 *
 * Controllers see the true arena state: no X-mirroring, no slot shuffling.
 */
class ExternalController {
public:
    virtual ~ExternalController() = default;

    /**
     * @brief Takes in the packet for this tick, like a bot decoding it.
     */
    virtual void observe(const ffi::ArenaState& state, const GameInfo& info) = 0;

    /**
     * @brief Whether the controller needs a decision for the packet it just observed.
     */
    [[nodiscard]] virtual bool wants_decision() const = 0;

    /**
     * @brief Writes the car's obs for the observed packet.
     *
     * @param out Must have room for the bot's `obs_size(state.num_cars)` floats.
     */
    virtual void build_obs(const ffi::ArenaState& state, float* out) const = 0;

    /**
     * @brief Takes the bot's inference result for the obs from `build_obs`.
     *
     * @param decision The bot's `decision_size()` floats for this car.
     * @param rng The env's RNG, for bots that sample.
     */
    virtual void set_decision(const float* decision, std::mt19937& rng) = 0;

    /**
     * @brief The car's controls for the observed packet, applied to the next physics tick.
     */
    [[nodiscard]] virtual ffi::CarControls tick(const ffi::ArenaState& state) = 0;
};

/**
 * @brief A bot from outside the framework, such as a port of a public RLBot bot.
 *
 * One instance is shared by every car it drives, across envs, so `infer` must be safe to call from several env
 * worker threads at once. Per-car state belongs in the controllers.
 */
class ExternalBot {
public:
    virtual ~ExternalBot() = default;

    [[nodiscard]] virtual std::string name() const = 0;

    /**
     * @brief Floats in one car's obs, in a match with `num_cars` cars.
     */
    [[nodiscard]] virtual size_t obs_size(size_t num_cars) const = 0;

    /**
     * @brief Floats in one car's inference result.
     */
    [[nodiscard]] virtual size_t decision_size() const = 0;

    /**
     * @brief Runs the bot on a batch of obs from matches with `num_cars` cars.
     *
     * @param obs `count` obs of `obs_size(num_cars)` floats each, back to back.
     * @param decisions Receives `count` results of `decision_size()` floats each.
     */
    virtual void infer(const float* obs, size_t count, size_t num_cars, float* decisions) = 0;

    /**
     * @brief Creates the controller for the car with this id, as the bot is when it joins a match.
     */
    [[nodiscard]] virtual std::unique_ptr<ExternalController> make_controller(uint32_t car_id) const = 0;
};

/**
 * @brief An external bot and the team it drives.
 */
struct ExternalSide {
    std::shared_ptr<ExternalBot> bot;
    ffi::Team team = ffi::Team::Orange;
};

} // namespace buta_ppo::external
