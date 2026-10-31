// buta-ppo/src/rl/actor_critic.hpp
#pragma once

#include <torch/torch.h>
#include <vector>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

namespace buta_ppo::rl {

/**
 * @brief Config params to define the MLP layout.
 * 
 * NOTE: User MUST fill out the layers. `obs_size` and `action_size` are filled in by the runner.
 */
struct ActorCriticConfig {
    int64_t obs_size;
    int64_t action_size;

    // Vectors to allow for custom layers.
    std::vector<int64_t> shared_layers;
    std::vector<int64_t> actor_layers;
    std::vector<int64_t> critic_layers;

    // A LayerNorm between each hidden layer and its ReLU. Stored in the model file, which the bot builds from.
    bool use_layer_norm = true;

    bool operator==(const ActorCriticConfig&) const = default;
};

/**
 * @brief What a saved model stores besides its weights: everything needed to rebuild the network and the obs builder
 * it was trained with, so a model file runs on its own (e.g. in the RLBot bot) without repeating the training config.
 */
struct ModelConfig {
    ActorCriticConfig network;
    int64_t max_players_per_team;
    int64_t ticks_per_step;

    bool operator==(const ModelConfig&) const = default;
};

/**
 * @brief The network section a parameter belongs to.
 */
enum class ParamSection { Shared, Actor, Critic };

/**
 * @brief The role of a parameter, used to choose its optimizer.
 */
enum class ParamKind {
    HiddenWeight, // Hidden-to-hidden Linear weight.
    IOWeight,     // Input-layer or output-head Linear weight.
    Vector        // Bias or LayerNorm parameter.
};

/**
 * @brief A parameter along with its section and role.
 */
struct ParamInfo {
    std::string name;
    torch::Tensor tensor;
    ParamSection section;
    ParamKind kind;
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

    // Names of the Linear weights fed directly by the raw obs.
    std::vector<std::string> input_weight_names_;

    // Helper method to build the libtorch neural network layers and connect them.
    torch::nn::Sequential build_block(int64_t in_size, const std::vector<int64_t>& sizes, bool use_ln);

public:
    explicit ActorCriticImpl(const ActorCriticConfig& config);

    /**
     * @brief Classifies every parameter by network section and role.
     *
     * @return Each parameter's name, tensor, section, and kind in `named_parameters()` order.
     */
    [[nodiscard]] std::vector<ParamInfo> parameter_layout() const;

    /**
     * @brief Loads weights written by `torch::save` into this network, after checking the file matches it.
     *
     * Unlike `torch::load`, which takes on the file's shapes without checking them, every tensor and layer in the
     * file must exist in this network with the same shape, and the other way around. So a checkpoint from a
     * different network config (layer sizes, layer norm, obs or action size) fails here instead of loading silently.
     * Values are copied into the existing tensors, which keep their devices and addresses.
     *
     * The `ModelConfig` stored by `save_model` is skipped, so this also loads models saved before configs were stored.
     *
     * @param path The file to load, as written by `save_model` or `torch::save` of an `ActorCritic`.
     * @throws std::runtime_error If the file can't be read or doesn't match, naming the first mismatch.
     */
    void load_weights(const std::string& path);

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
     * @brief Samples actions from the policy dist without running the critic. Used for opponents that aren't trained.
     * 
     * @param obs The batched obs tensor.
     * @param action_masks The float mask tensor indicating valid env actions.
     * @return The sampled actions.
     */
    torch::Tensor sample_action(torch::Tensor obs, torch::Tensor action_masks);

    /**
     * @brief The most likely valid actions, without running the critic. How the bot plays outside training.
     *
     * @param obs The batched obs tensor.
     * @param action_masks The float mask tensor indicating valid env actions.
     * @return The best actions.
     */
    torch::Tensor best_action(torch::Tensor obs, torch::Tensor action_masks);

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

/**
 * @brief Saves a model's weights along with its `ModelConfig`.
 *
 * @param model The network to save.
 * @param config The config `model` was built and trained with.
 * @param path The file to write.
 */
void save_model(const ActorCritic& model, const ModelConfig& config, const std::string& path);

/**
 * @brief Reads the `ModelConfig` stored in a model file, without loading the weights.
 *
 * @param path A file written by `save_model`.
 * @return The stored config, or nullopt for a model saved before configs were stored.
 * @throws std::runtime_error If the file can't be read, or its config is incomplete or from a newer version.
 */
[[nodiscard]] std::optional<ModelConfig> read_model_config(const std::string& path);

/**
 * @brief Describes a `ModelConfig` on one line, for logs and error messages.
 */
[[nodiscard]] std::string describe_model_config(const ModelConfig& config);

} // namespace buta_ppo::rl 