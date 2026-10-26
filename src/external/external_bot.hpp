// buta-ppo/src/external/external_bot.hpp
#pragma once

#include "buta_ppo/ffi.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

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
 * the match across goals. Evaluation's lockstep play instead gives every game fresh controllers, so games repeat.
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

    /**
     * @brief Packets after the one a decision is asked for until the controller first uses it, so the decision can
     * arrive late. Evaluation uses it to batch the inference of several packets at once.
     *
     * With a delay of d, the env may call `request_decision` at the decision's packet instead of `set_decision`, then
     * `set_decision` up to d packets later, before that packet's `tick`. Decisions are set in the order they were
     * asked for. The game plays out the same either way. 0, the default, means `set_decision` right away.
     */
    [[nodiscard]] virtual int decision_delay() const { return 0; }

    /**
     * @brief Takes note that the observed packet's decision was asked for, to be set later (see `decision_delay`).
     * Called before the packet's `tick`.
     */
    virtual void request_decision() {}

    /**
     * @brief The fewest packets between two of the car's decisions. Sizes evaluation's batches: a car asks for at most
     * one decision per this many packets.
     */
    [[nodiscard]] virtual int decision_interval() const { return 1; }
};

/**
 * @brief Runs a bot's inference in evaluation, batched across every game deciding on the same tick.
 *
 * Built once for a set of evaluation games. Batches are per car count, since that sets the obs size. Batches of
 * different car counts may run at the same time: fill and `launch` each, then `wait` for each.
 */
class ExternalBatchRunner {
public:
    virtual ~ExternalBatchRunner() = default;

    /**
     * @brief Room for the largest batch of obs from matches with `num_cars` cars, which the caller fills before `launch`.
     */
    [[nodiscard]] virtual float* obs(size_t num_cars) = 0;

    /**
     * @brief Starts inference on the first `count` obs in `obs(num_cars)`. May return before it's done.
     */
    virtual void launch(size_t num_cars, size_t count) = 0;

    /**
     * @brief Waits for the last `launch` with `num_cars` cars and returns its `count` decisions, back to back.
     */
    [[nodiscard]] virtual const float* wait(size_t num_cars) = 0;
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

    /**
     * @brief What runs the bot's inference. Bots that return the same model give the same decisions for the same obs,
     * so evaluation batches their cars together (e.g. one network played with different options).
     */
    [[nodiscard]] virtual const void* model() const { return this; }

    /**
     * @brief The bot's inference for evaluation, batched across games.
     *
     * @param max_batch The most obs a batch can have, per car count.
     * @return By default, a runner that calls `infer`.
     */
    [[nodiscard]] virtual std::unique_ptr<ExternalBatchRunner> make_batch_runner(const std::map<size_t, size_t>& max_batch);
};

/**
 * @brief The default `ExternalBatchRunner`: `infer` on the calling thread, at `launch`.
 */
class EagerBatchRunner final : public ExternalBatchRunner {
public:
    EagerBatchRunner(ExternalBot& bot, const std::map<size_t, size_t>& max_batch) : bot_(bot) {
        for (const auto& [num_cars, count] : max_batch) {
            Buffers& b = buffers_[num_cars];
            b.obs.resize(count * bot.obs_size(num_cars));
            b.decisions.resize(count * bot.decision_size());
        }
    }

    [[nodiscard]] float* obs(size_t num_cars) override { return buffers_.at(num_cars).obs.data(); }
    void launch(size_t num_cars, size_t count) override {
        Buffers& b = buffers_.at(num_cars);
        bot_.infer(b.obs.data(), count, num_cars, b.decisions.data());
    }
    [[nodiscard]] const float* wait(size_t num_cars) override { return buffers_.at(num_cars).decisions.data(); }

private:
    struct Buffers {
        std::vector<float> obs;
        std::vector<float> decisions;
    };
    ExternalBot& bot_;
    std::map<size_t, Buffers> buffers_;
};

inline std::unique_ptr<ExternalBatchRunner> ExternalBot::make_batch_runner(const std::map<size_t, size_t>& max_batch) {
    return std::make_unique<EagerBatchRunner>(*this, max_batch);
}

/**
 * @brief An external bot and the team it drives.
 */
struct ExternalSide {
    std::shared_ptr<ExternalBot> bot;
    ffi::Team team = ffi::Team::Orange;
};

} // namespace buta_ppo::external
