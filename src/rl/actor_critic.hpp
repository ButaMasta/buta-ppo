// buta-ppo/src/rl/actor_critic.hpp
#pragma once

#include <torch/torch.h>
#include <vector>
#include <cstdint>
#include <tuple>

namespace buta_ppo::rl {

/**
 * @brief Config params to define the MLP layout.
 * 
 * NOTE: User MUST fill out these fields. Default values are intentionally omitted.
 */
struct ActorCriticConfig {
    int64_t obs_size;
    int64_t action_size;

    // Vectors to allow for custom layers.
    std::vector<int64_t> shared_layers;
    std::vector<int64_t> actor_layers;
    std::vector<int64_t> critic_layers;

    bool use_layer_norm;
};

/**
 * @brief LibTorch module implementation for the network.
 */
class ActorCriticImpl : public torch::nn::Module {
private:
    // Shared initial layers.
    torch::nn::Sequential shared_mlp_{nullptr};

    // Separate branching or initial layers.
    torch::nn::Sequential actor_mlp_{nullptr};
    torch::nn::Sequential critic_mlp_{nullptr};

    // Final output layers.
    torch::nn::Linear actor_head_{nullptr};
    torch::nn::Linear critic_head_{nullptr};

    // Helper method to build the libtorch neural network layers and connect them.
    torch::nn::Sequential build_block(int64_t in_size, const std::vector<int64_t>& sizes, bool use_ln);

public:
    explicit ActorCriticImpl(const ActorCriticConfig& config);

    /**
     * @brief Performs a forward pass through the shared, actor, and critic networks. 
     * 
     * @param obs The batched obs tensor.
     * @return A tuple containing the unnormalized action logits and the state value predictions.
     */
    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor obs);

    /**
     * @brief Performs a forward pass through only the shared and actor networks. 
     * 
     * @param obs The batched obs tensor.
     * @return The unnormalized action logits.
     */
    [[nodiscard]] torch::Tensor forward_actor(torch::Tensor obs);

    /**
     * @brief Performs a forward pass through only the shared and critic networks. 
     * 
     * @param obs The batched obs tensor.
     * @return The unnormalized action logits.
     */
    [[nodiscard]] torch::Tensor forward_critic(torch::Tensor obs);

    /**
     * @brief Samples actions from the policy dist and estimates state values during rollout.
     * 
     * @param obs The batched obs tensor.
     * @param action_masks The bool mask tensor indicating valid env actions.
     * @return A tuple containing sampled actions, their log probs, and the state values.
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_action_and_value(
        torch::Tensor obs,
        torch::Tensor action_masks
    );

    /**
     * @brief Evaluates historical actions to compute log probs and policy entropy during back prop.
     * 
     * @param obs The batched obs tensor.
     * @param actions The tensor of actions previously selected by the policy.
     * @param action_masks The bool mask tensor indicating valid env actions.
     * @return A tuple containing the log probs, dist entropy, and state values. 
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate_actions(
        torch::Tensor obs,
        torch::Tensor actions,
        torch::Tensor action_masks
    );
};

// Wrap class in torch module.
TORCH_MODULE(ActorCritic);

} // namespace buta_ppo::rl 