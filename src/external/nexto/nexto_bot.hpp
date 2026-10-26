// Derived from Necto (https://github.com/Rolv-Arild/Necto)
// by Rolv-Arild and the Necto contributors.
// Translated to C++ for buta-ppo. Included under the MIT License
// with permission; see THIRD_PARTY_NOTICES.md.

// buta-ppo/src/external/nexto/nexto_bot.hpp

/*
*   Port of Nexto, the RLGym community bot by Rolv, Soren and contributors (https://github.com/Rolv-Arild/Necto),
*   as it plays in the RLBot botpack: `bot.py`, `agent.py` and `nexto_obs.py`, with the state decoding of
*   rlgym_compat 1.0.2, which that version runs on. The model is the botpack's TorchScript `nexto-model.pt`.
*
*   RLBot values are taken from RocketSim as follows:
*       has_wheel_contact -> is_on_ground (RocketSim: 3+ wheels touching)
*       double_jumped     -> has_double_jumped || has_flipped (RocketSim tracks a dodge separately)
*       boost pad active  -> cooldown == 0, mapped by position to RLBot's pad order
*       car rotation      -> RocketSim's forward/up, through the original's quaternion round trip
*       is_kickoff_pause  -> the env's GameInfo (from "go" until the first touch; false during the countdown)
*/

#pragma once

#include "external/external_bot.hpp"
#include "nexto_net.hpp"

#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <map>

namespace buta_ppo::external {

/**
 * @brief Options of the botpack Nexto, with its defaults.
 */
struct NextoOptions {
    // How random its actions are: 1 = best action, 0.5 = sampled from the policy, 0 = uniform, -1 = worst action.
    float beta = 1.0f;
    // The kickoff taker plays a scripted speedflip kickoff.
    bool hardcoded_kickoffs = true;
    // Actions are sampled (beta 0.5) during the kickoff pause.
    bool stochastic_kickoffs = true;

    // Where the network runs. On CUDA it runs without TF32, which changes its choices.
    torch::Device device = torch::kCUDA;
    // Runs the original TorchScript file (CPU only) instead of the native network. The reference for checks.
    bool torchscript = false;
};

namespace nexto {

inline constexpr int TICK_SKIP = 8;
inline constexpr size_t NUM_ACTIONS = 90;
inline constexpr size_t ACTION_SIZE = 8; // throttle, steer, pitch, yaw, roll, jump, boost, handbrake
inline constexpr size_t QUERY_SIZE = 32;
inline constexpr size_t ENTITY_SIZE = 24;
inline constexpr size_t NUM_PADS = 34;
inline constexpr size_t MAX_CARS = std::extent_v<decltype(ffi::ArenaState::cars)>;

using Action = std::array<float, ACTION_SIZE>;

/**
 * @brief Entities in the obs: every car, the ball and the boost pads.
 */
[[nodiscard]] constexpr size_t num_entities(size_t num_cars) { return num_cars + 1 + NUM_PADS; }

/**
 * @brief `agent.py`'s action lookup table.
 */
[[nodiscard]] const std::array<Action, NUM_ACTIONS>& action_table();

} // namespace nexto

/**
 * @brief The Nexto model, shared by every car Nexto drives.
 *
 * Runs `nexto::NextoNet`, built from the TorchScript file's weights, on the options' device. On CUDA, obs and logits
 * pass through pinned staging buffers and the network runs on its own stream, never the default one.
 *
 * `with_options` makes Nexto with other options (e.g. another beta) on the same network. In evaluation, the cars of
 * every such bot are batched together, through CUDA graphs on CUDA (see `make_batch_runner`).
 */
class NextoBot final : public ExternalBot {
public:
    NextoBot(const std::string& model_path, const NextoOptions& options);

    /**
     * @brief Nexto with other options, sharing this one's network. The network's options (`device`, `torchscript`)
     * stay this one's.
     */
    [[nodiscard]] std::shared_ptr<NextoBot> with_options(const NextoOptions& options) const;

    [[nodiscard]] std::string name() const override { return "Nexto"; }
    [[nodiscard]] size_t obs_size(size_t num_cars) const override {
        return nexto::QUERY_SIZE + nexto::num_entities(num_cars) * nexto::ENTITY_SIZE;
    }
    // The action logits.
    [[nodiscard]] size_t decision_size() const override { return nexto::NUM_ACTIONS; }

    void infer(const float* obs, size_t count, size_t num_cars, float* decisions) override;
    [[nodiscard]] std::unique_ptr<ExternalController> make_controller(uint32_t car_id) const override;

    [[nodiscard]] const void* model() const override { return model_.get(); }

    /**
     * @brief On CUDA, runs the network through CUDA graphs: per car count, one per power-of-2 batch up to the largest,
     * each uploading the obs, running the network (captured without TF32, like `infer`) and downloading the logits.
     * Batches are padded up to their graph's size, and each car count has its own stream so they run at the same time.
     * Otherwise `infer`.
     */
    [[nodiscard]] std::unique_ptr<ExternalBatchRunner> make_batch_runner(const std::map<size_t, size_t>& max_batch) override;

    [[nodiscard]] const NextoOptions& options() const { return options_; }

private:
    // The network and its inference, shared by every NextoBot made `with_options`.
    struct Model {
        torch::Device device = torch::kCPU;
        bool torchscript = false;
        torch::jit::Module module; // The TorchScript file: the weights' source, and the `torchscript` reference.
        nexto::NextoNet net;
        std::mutex mutex; // Serializes `infer` across env workers.

        // CUDA only. Staging buffers grow to the largest batch seen.
        std::optional<c10::cuda::CUDAStream> stream;
        torch::Tensor host_obs;    // Pinned.
        torch::Tensor device_obs;
        torch::Tensor host_logits; // Pinned.
    };
    class GraphRunner;

    std::shared_ptr<Model> model_;
    NextoOptions options_;

    NextoBot(std::shared_ptr<Model> model, const NextoOptions& options) : model_(std::move(model)), options_(options) {}

    [[nodiscard]] torch::Tensor run_torchscript(const torch::Tensor& query, const torch::Tensor& entities);
    [[nodiscard]] torch::Tensor run_cuda(const float* obs, int64_t count, int64_t entities);
};

/**
 * @brief One Nexto car: `bot.py`'s per-packet `get_output` and `maybe_do_kickoff`, and the state it decodes.
 *
 * `bot.py` builds its obs when its tick counter wraps to 1 and applies the chosen action from tick 7, so an action
 * takes effect 6 ticks after the packet it was chosen from and lasts 8 ticks. The counter runs from when the bot
 * joined and is never reset, so its phase at a kickoff is whatever the play before left it at.
 */
class NextoController final : public ExternalController {
public:
    NextoController(uint32_t car_id, const NextoOptions& options);

    void observe(const ffi::ArenaState& state, const GameInfo& info) override;
    [[nodiscard]] bool wants_decision() const override { return update_action_; }
    void build_obs(const ffi::ArenaState& state, float* out) const override;
    void set_decision(const float* decision, std::mt19937& rng) override;
    [[nodiscard]] ffi::CarControls tick(const ffi::ArenaState& state) override;

    // An action is first used at the 7th tick after the packet it was chosen from (see the class comment).
    [[nodiscard]] int decision_delay() const override { return nexto::TICK_SKIP - 2; }
    void request_decision() override;
    // One decision per tick cycle.
    [[nodiscard]] int decision_interval() const override { return nexto::TICK_SKIP; }

    // For checks against the Python bot.
    [[nodiscard]] size_t last_action_index() const { return last_action_index_; }
    [[nodiscard]] float last_beta() const { return last_beta_; }

private:
    uint32_t car_id_;
    NextoOptions options_;
    GameInfo info_; // The observed packet's.

    // bot.py's `ticks`, `update_action`, `action` (also the previous action in the obs) and `controls`, as
    // `initialize_agent` sets them.
    int ticks_ = nexto::TICK_SKIP;
    bool update_action_ = true;
    nexto::Action action_{};
    ffi::CarControls controls_{};

    // bot.py's `kickoff_index`: -1 before a kickoff is checked, -2 when not the kickoff taker.
    int kickoff_index_ = -1;

    // rlgym_compat's `_on_ground_ticks`: ticks since each car last had wheel contact.
    std::array<int, nexto::MAX_CARS> ticks_off_ground_{};

    size_t last_action_index_ = 0;
    float last_beta_ = 0.0f;

    // A decision asked for but not yet set (`request_decision`): the beta of its packet, and whether a scripted kickoff
    // has replaced the action since, which a decision set right away would have been replaced by too.
    bool pending_ = false;
    float pending_beta_ = 0.0f;
    bool pending_overwritten_ = false;

    /**
     * @brief rlgym_compat's `on_ground`: wheel contact, or at most 6 ticks since the last contact.
     */
    [[nodiscard]] bool on_ground(uint32_t car_id, const ffi::ArenaState& state) const;

    /**
     * @brief bot.py's `maybe_do_kickoff`.
     */
    void maybe_do_kickoff(const ffi::ArenaState& state);

    [[nodiscard]] bool is_kickoff_taker(const ffi::ArenaState& state) const;
};

} // namespace buta_ppo::external
