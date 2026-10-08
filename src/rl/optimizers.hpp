// buta-ppo/src/rl/optimizers.hpp
#pragma once

#include "actor_critic.hpp"

#include <torch/torch.h>

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
 * @brief An optimizer whose step only issues device work, so it can be captured in a CUDA graph and replayed.
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
 * @return The optimizers, which together cover every parameter exactly once.
 * @throws std::invalid_argument If Muon is set on a non-hidden group or a learning rate isn't positive.
 */
[[nodiscard]] std::vector<std::unique_ptr<GraphSafeOptimizer>> build_optimizers(
    const std::vector<ParamInfo>& layout,
    const OptimizerConfig& config
);

} // namespace buta_ppo::rl
