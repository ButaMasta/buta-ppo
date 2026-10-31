// buta-ppo/src/rl/optimizers.hpp
#pragma once

#include "actor_critic.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace buta_ppo::rl {

/**
 * @brief The update rule applied to a parameter group.
 */
enum class OptimizerType { Adam, AdamW, Muon };

/**
 * @brief How Muon scales its learning rate for each matrix shape.
 */
enum class MuonLrScale {
    MatchAdamW, // 0.2 * sqrt(max(rows, cols)): matches AdamW's update RMS so AdamW-scale lrs carry over.
    Original    // sqrt(max(1, rows / cols)): the original scaling, which expects much larger lrs (~0.02).
};

/**
 * @brief The optimizer and hyperparameters for one parameter group.
 */
struct OptimizerSpec {
    OptimizerType type = OptimizerType::Adam;
    float lr = 3e-4f;
    float weight_decay = 0.0f; // Adam: L2 added to the gradient. AdamW/Muon: decoupled decay.

    // Adam / AdamW.
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;

    // Muon. Defaults match torch.optim.Muon.
    float momentum = 0.95f;
    bool nesterov = true;
    int ns_steps = 5;
    MuonLrScale muon_lr_scale = MuonLrScale::MatchAdamW;

    bool operator==(const OptimizerSpec&) const = default;
};

/**
 * @brief The optimizer specs for one network section, split by parameter role.
 */
struct SectionOptimizer {
    OptimizerSpec hidden_weights; // Hidden-to-hidden Linear weights. The only group Muon may be used on.
    OptimizerSpec io_weights;     // Input-layer and output-head Linear weights.
    OptimizerSpec vectors;        // Biases and LayerNorm parameters.

    /**
     * @brief Sets the learning rate of all three groups in this section.
     *
     * @param lr The learning rate.
     * @return This section, for chaining.
     */
    SectionOptimizer& with_lr(float lr) {
        hidden_weights.lr = lr;
        io_weights.lr = lr;
        vectors.lr = lr;
        return *this;
    }
};

/**
 * @brief The optimizer setup for the whole network, per section (shared / actor / critic).
 *
 * Start from a preset and adjust individual groups as needed, e.g.
 * `auto opt = OptimizerConfig::muon(3e-4f, 3e-4f, 0.01f); opt.critic.with_lr(1e-4f);`
 */
struct OptimizerConfig {
    SectionOptimizer shared;
    SectionOptimizer actor;
    SectionOptimizer critic;

    /**
     * @brief Adam on every parameter.
     */
    [[nodiscard]] static OptimizerConfig adam(float lr) {
        const OptimizerSpec spec{.type = OptimizerType::Adam, .lr = lr};
        const SectionOptimizer section{spec, spec, spec};
        return {section, section, section};
    }

    /**
     * @brief AdamW on every parameter, with no weight decay on biases and LayerNorm parameters.
     */
    [[nodiscard]] static OptimizerConfig adamw(float lr, float weight_decay) {
        const OptimizerSpec weights{.type = OptimizerType::AdamW, .lr = lr, .weight_decay = weight_decay};
        const OptimizerSpec vectors{.type = OptimizerType::AdamW, .lr = lr, .weight_decay = 0.0f};
        const SectionOptimizer section{weights, weights, vectors};
        return {section, section, section};
    }

    /**
     * @brief Muon on hidden weights and AdamW everywhere else, with no weight decay on biases and LayerNorm parameters.
     */
    [[nodiscard]] static OptimizerConfig muon(float muon_lr, float adamw_lr, float weight_decay) {
        const OptimizerSpec hidden{.type = OptimizerType::Muon, .lr = muon_lr, .weight_decay = weight_decay};
        const OptimizerSpec io{.type = OptimizerType::AdamW, .lr = adamw_lr, .weight_decay = weight_decay};
        const OptimizerSpec vectors{.type = OptimizerType::AdamW, .lr = adamw_lr, .weight_decay = 0.0f};
        const SectionOptimizer section{hidden, io, vectors};
        return {section, section, section};
    }
};

/**
 * @brief Scales every learning rate of `PPOConfig::optimizer` by a factor that changes with the lifetime steps.
 *
 * Holds `start_scale` up to `start_steps`, moves to `end_scale` by `end_steps` along `shape`, and holds `end_scale`
 * after. The default is 1 throughout. The runner sets the scale before each update, and the captured training graph
 * reads it, so it changes without a restart, and a resumed run picks it up from its step count. Scaling every group
 * keeps the ratios between them (Muon's lr against AdamW's, a critic set lower).
 *
 * `LrSchedule::exponential(7'000'000'000, 12'000'000'000, 1.0f, 0.25f)` halves the learning rates twice between 7B and
 * 12B steps, at an even pace.
 */
struct LrSchedule {
    enum class Shape {
        Constant,    // `start_scale` throughout.
        Linear,      // A straight line from `start_scale` to `end_scale`.
        Exponential, // A geometric one: equal stretches of steps take off equal fractions.
        Cosine       // Half a cosine: slow at both ends, fastest in the middle.
    };

    Shape shape = Shape::Constant;
    int64_t start_steps = 0;
    int64_t end_steps = 0;
    float start_scale = 1.0f;
    float end_scale = 1.0f;

    [[nodiscard]] static LrSchedule linear(int64_t start_steps, int64_t end_steps, float start_scale, float end_scale) {
        return {Shape::Linear, start_steps, end_steps, start_scale, end_scale};
    }
    [[nodiscard]] static LrSchedule exponential(int64_t start_steps, int64_t end_steps, float start_scale, float end_scale) {
        return {Shape::Exponential, start_steps, end_steps, start_scale, end_scale};
    }
    [[nodiscard]] static LrSchedule cosine(int64_t start_steps, int64_t end_steps, float start_scale, float end_scale) {
        return {Shape::Cosine, start_steps, end_steps, start_scale, end_scale};
    }

    /**
     * @brief The scale at `steps` lifetime steps.
     */
    [[nodiscard]] float scale_at(int64_t steps) const;

    /**
     * @throws std::invalid_argument If a scale isn't positive and finite, or a changing schedule doesn't end after it
     *         starts.
     */
    void validate() const;

    bool operator==(const LrSchedule&) const = default;
};

/**
 * @brief An optimizer whose step only issues device work, so it can be captured in a CUDA graph and replayed.
 *
 * Its learning rates are multiplied by a device scalar (see `LrSchedule`), read at every step, so a captured graph
 * picks up a new scale on its next replay.
 */
class GraphSafeOptimizer {
public:
    virtual ~GraphSafeOptimizer() = default;

    /**
     * @brief Applies one update using the current gradients of this optimizer's parameters.
     */
    virtual void step() = 0;

    /**
     * @brief All state tensors, keyed `<state>.<parameter name>`, for checkpointing and warmup backup.
     */
    [[nodiscard]] virtual std::vector<std::pair<std::string, torch::Tensor>> named_state() const = 0;

    /**
     * @brief A one-line summary of the optimizer type, hyperparameters, and parameters.
     */
    [[nodiscard]] virtual std::string describe() const = 0;
};

/**
 * @brief Assigns every parameter its spec from the config and builds one optimizer per distinct spec.
 *
 * @param layout The classified parameters of the network.
 * @param config The optimizer config.
 * @param lr_scale A 0-dim float32 tensor on the parameters' device that every learning rate is multiplied by. Its
 *                 value is read at every step; the tensor itself must stay the same one.
 * @return The optimizers, which together cover every parameter exactly once.
 * @throws std::invalid_argument If Muon is set on a non-hidden group or a learning rate isn't positive.
 */
[[nodiscard]] std::vector<std::unique_ptr<GraphSafeOptimizer>> build_optimizers(
    const std::vector<ParamInfo>& layout,
    const OptimizerConfig& config,
    const torch::Tensor& lr_scale
);

} // namespace buta_ppo::rl
