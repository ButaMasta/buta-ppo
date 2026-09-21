// buta-ppo/src/rl/actor_critic.hpp

#include <torch/torch.h>

namespace buta_ppo::rl {

// User MUST fill out these fields. Default values are intentionally omitted.
struct ActorCriticConfig {
    int64_t obs_size;
    int64_t action_size;

    // Vectors to allow for custom layers.
    std::vector<int64_t> shared_layers;
    std::vector<int64_t> actor_layers;
    std::vector<int64_t> critic_layers;

    bool use_layer_norm;
};

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

    torch::nn::Sequential build_block(int64_t in_size, const std::vector<int64_t>& sizes, bool use_ln);

public:
    explicit ActorCriticImpl(const ActorCriticConfig& config);

    // Takes in the observation tensor and returns {logits, values}
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor obs);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_action_and_value(
        torch::Tensor obs,
        torch::Tensor action_masks
    );
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate_actions(
        torch::Tensor obs,
        torch::Tensor actions,
        torch::Tensor action_masks
    );
};

// Wrap class in torch module.
TORCH_MODULE(ActorCritic);

}; // namespace buta_ppo::rl 