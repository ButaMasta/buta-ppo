// buta-ppo/src/rl/actor_critic.cpp
#include "actor_critic.hpp"

namespace buta_ppo::rl {

torch::nn::Sequential ActorCriticImpl::build_block(int64_t in_size, const std::vector<int64_t>& sizes, bool use_ln) {
    if (sizes.empty()) return nullptr;

    torch::nn::Sequential seq;
    int64_t current_in = in_size;

    for (int64_t out_size : sizes) {
        auto linear = torch::nn::Linear(current_in, out_size);

        torch::nn::init::orthogonal_(linear->weight, std::sqrt(2.0));
        torch::nn::init::constant_(linear->bias, 0.0);

        seq->push_back(linear);
        if (use_ln) {
            seq->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({out_size})));
        }
        seq->push_back(torch::nn::ReLU());
        current_in = out_size;
    }
    return seq;
}

ActorCriticImpl::ActorCriticImpl(const ActorCriticConfig& config) {
    // Build shared layers.
    int64_t shared_out_size = config.obs_size;
    if (!config.shared_layers.empty()) {
        shared_mlp_ = register_module("shared_mlp", build_block(config.obs_size, config.shared_layers, config.use_layer_norm));
        shared_out_size = config.shared_layers.back();
    }

    // Build actor layers.
    int64_t actor_out_size = shared_out_size;
    if (!config.actor_layers.empty()) {
        actor_mlp_ = register_module("actor_mlp", build_block(shared_out_size, config.actor_layers, config.use_layer_norm));
        actor_out_size = config.actor_layers.back();
    }

    // Build critic layers.
    int64_t critic_out_size = shared_out_size;
    if (!config.critic_layers.empty()) {
        critic_mlp_ = register_module("critic_mlp", build_block(shared_out_size, config.critic_layers, config.use_layer_norm));
        critic_out_size = config.critic_layers.back();
    }

    actor_head_ = register_module("actor_head", torch::nn::Linear(actor_out_size, config.action_size));
    torch::nn::init::orthogonal_(actor_head_->weight, 0.01);
    torch::nn::init::constant_(actor_head_->bias, 0.0);

    critic_head_ = register_module("critic_head", torch::nn::Linear(critic_out_size, 1));
    torch::nn::init::orthogonal_(critic_head_->weight, 1.0);
    torch::nn::init::constant_(critic_head_->bias, 0.0);
}

std::tuple<torch::Tensor, torch::Tensor> ActorCriticImpl::forward(torch::Tensor obs) {
    torch::Tensor shared_features = obs;
    if (shared_mlp_) {
        shared_features = shared_mlp_->forward(shared_features);
    }

    torch::Tensor actor_features = shared_features;
    if (actor_mlp_) {
        actor_features = actor_mlp_->forward(actor_features);
    }

    torch::Tensor critic_features = shared_features;
    if (critic_mlp_) {
        critic_features = critic_mlp_->forward(critic_features);
    }

    torch::Tensor logits = actor_head_->forward(actor_features);
    torch::Tensor values = critic_head_->forward(critic_features);

    return { logits, values };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ActorCriticImpl::get_action_and_value(
    torch::Tensor obs,
    torch::Tensor action_masks
) {
    torch::NoGradGuard no_grad;

    auto [logits, values] = forward(obs);

    // Apply negative penalty to invalid action to make their softmax probs near zero.
    // NOTE: action_masks is of float type here so the cast is necessary.
    torch::Tensor masked_logits = torch::where(
        action_masks.to(torch::kBool),
        logits,
        torch::tensor(-1e8f, logits.options())
    );

    torch::Tensor probs = torch::softmax(masked_logits, -1);
    torch::Tensor actions = torch::multinomial(probs, 1).squeeze(-1);

    torch::Tensor log_probs = torch::log_softmax(masked_logits, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1);

    return { actions, log_probs, values };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ActorCriticImpl::evaluate_actions(
    torch::Tensor obs,
    torch::Tensor actions,
    torch::Tensor action_masks
) {
    auto [logits, values] = forward(obs);

    // NOTE: action_masks is of bool type here so the cast is not needed.
    torch::Tensor masked_logits = torch::where(
        action_masks,
        logits,
        torch::tensor(-1e8f, logits.options())
    );

    torch::Tensor log_softmax_logits = torch::log_softmax(masked_logits, -1);
    torch::Tensor log_probs = log_softmax_logits.gather(-1, actions.unsqueeze(-1)).squeeze(-1);

    torch::Tensor probs = torch::softmax(masked_logits, -1);
    torch::Tensor p_log_p = probs * log_softmax_logits;

    p_log_p = torch::nan_to_num(p_log_p, 0.0f);
    torch::Tensor entropy = -p_log_p.sum(-1);

    return { log_probs, entropy, values };
}

} // namespace buta_ppo::rl 