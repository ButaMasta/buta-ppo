// buta-ppo/src/rl/ppo_trainer.hpp
#pragma once

#include "actor_critic.hpp"
#include "optimizers.hpp"

#include <torch/torch.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAGraph.h>

#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

// Forward declaration.
namespace buta_ppo::rl { class RolloutBuffer; }

namespace buta_ppo::rl {

/**
 * @brief The config for the PPO hyperparameters and values.
 */
struct PPOConfig {
    float clip_ratio = 0.2f;
    float value_coef = 0.5f;
    float entropy_coef = 0.035f;
    float target_kl = 0.015f;
    float max_grad_norm = 0.5f;

    int epochs = 3;
    int mini_batch_size = 1024; // Highly hardware-dependant. This is set by runner config initialization.

    float gae_gamma = 0.99f;
    float gae_lambda = 0.95f;

    // Optimizer per network section and parameter role. See `OptimizerConfig` presets.
    OptimizerConfig optimizer = OptimizerConfig::adam(3e-4f);
};

/**
 * @brief Enforces BFloat16 mixed-precision within scoped execution blocks.
 */
struct BFloat16AutocastGuard {
    bool prev_enabled;
    c10::ScalarType prev_dtype;

    BFloat16AutocastGuard() {
        prev_enabled = at::autocast::is_autocast_enabled(torch::kCUDA);
        prev_dtype = at::autocast::get_autocast_dtype(torch::kCUDA);

        at::autocast::set_autocast_enabled(torch::kCUDA, true);
        at::autocast::set_autocast_dtype(torch::kCUDA, torch::kBFloat16);
    }

    ~BFloat16AutocastGuard() {
        at::autocast::set_autocast_enabled(torch::kCUDA, prev_enabled);
        at::autocast::set_autocast_dtype(torch::kCUDA, prev_dtype);

        at::autocast::clear_cache();
    }
};

/**
 * @brief Core training class that encapsulates the model and updating it according to collected experience.
 */
class PPOTrainer {
private:
    PPOConfig config_;
    ActorCritic actor_critic_;
    torch::Device device_;

    std::vector<std::string> param_names_;
    std::vector<torch::Tensor> params_;

    // One graph-safe optimizer per distinct spec. Together they cover every parameter exactly once.
    std::vector<std::unique_ptr<GraphSafeOptimizer>> optimizers_;

    // Reused grad handle list for clipping. Grads are re-created by each backward (set_to_none).
    std::vector<torch::Tensor> grads_;

    // Mini-batch shuffling.
    torch::Tensor batch_indices_;

    // CUDA Graph buffers.
    at::cuda::CUDAGraph graph_;
    bool graph_captured_ = false;

    // Graph inputs.
    torch::Tensor static_mb_obs_;
    torch::Tensor static_mb_actions_;
    torch::Tensor static_mb_action_masks_;
    torch::Tensor static_mb_old_log_probs_;
    torch::Tensor static_mb_advantages_;
    torch::Tensor static_mb_returns_;

    // Graph outputs.
    torch::Tensor static_policy_loss_;
    torch::Tensor static_approx_kl_;
    torch::Tensor static_clip_fraction_;
    torch::Tensor static_value_loss_;
    torch::Tensor static_entropy_;

    // An implementations of torch utils' clip grad norm that works with CUDA graphs.
    void graph_safe_clip_grad_norm(const std::vector<torch::Tensor>& grads, float max_norm);

    // Every optimizer's state tensors, keyed `<state>.<parameter name>`.
    [[nodiscard]] std::vector<std::pair<std::string, torch::Tensor>> optimizer_state() const;

public:
    PPOTrainer(PPOConfig config, ActorCritic actor_critic, torch::Device device);

    /**
     * @brief Executes PPO optimization loop over collected rollouts.
     * 
     * @param buffer The Rollout buffer containing all the data for training.
     * @return Metrics averaged over minibatches: losses, entropy, approx KL, and clip fraction.
     */
    [[nodiscard]] std::unordered_map<std::string, float> train_step(const RolloutBuffer& buffer);

    /**
     * @brief Saves the state of every optimizer, keyed by state and parameter name.
     *
     * @param path The file to write the optimizer state to.
     */
    void save_optimizer(const std::string& path) const;

    /**
     * @brief Loads optimizer state written by `save_optimizer` into the existing state tensors.
     *
     * If the file can't be read or doesn't match the network, the current state is left untouched.
     *
     * @param path The file to read the optimizer state from.
     * @return Whether the optimizer state was restored.
     */
    bool load_optimizer(const std::string& path);

    /**
     * @brief The core compute during consumption designed for CUDA graph capture and replay.
     */
    void execute_minibatch_graph_logic();
};

} // namespace buta_ppo::rl