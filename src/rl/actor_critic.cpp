// buta-ppo/src/rl/actor_critic.cpp
#include "actor_critic.hpp"

#include <ATen/autocast_mode.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace buta_ppo::rl {

namespace {

// Applies negative penalty to invalid actions to make their softmax probs near zero.
// NOTE: action_masks is of float type during rollout so the cast is necessary.
torch::Tensor mask_logits(const torch::Tensor& logits, const torch::Tensor& action_masks) {
    return torch::where(
        action_masks.to(torch::kBool),
        logits,
        torch::tensor(-1e8f, logits.options())
    );
}

// Sample with the exponential race: argmax(p / E) with E ~ Exp(1) picks index i with probability p_i.
// Unlike multinomial it never synchronizes with the host, so it can be captured in a CUDA graph.
// Masked actions have p = 0 and never win. E is clamped above 0 so they can't become 0 / 0 = NaN.
torch::Tensor sample_masked(const torch::Tensor& masked_logits) {
    torch::Tensor probs = torch::softmax(masked_logits, -1);
    torch::Tensor noise = torch::empty_like(probs).exponential_().clamp_min_(std::numeric_limits<float>::min());
    return (probs / noise).argmax(-1);
}

/**
 * @brief Turns CUDA autocast off for its scope, then back on.
 */
struct AutocastOffGuard {
    AutocastOffGuard() {
        at::autocast::set_autocast_enabled(torch::kCUDA, false);
    }

    ~AutocastOffGuard() {
        at::autocast::set_autocast_enabled(torch::kCUDA, true);
    }
};


/**
 * @brief Runs a block's layers in order, like seq->forward(x), but it ensures that when autocast is on and x is bf16 to run layernorm in bf16 as well.
 *
 * Autocast runs LayerNorm in fp32, which casts every activation to fp32 and back. 
 * The CUDA kernel accumulates in fp32 even for bf16 input (`acc_type<BFloat16> = float`, see `aten/src/ATen/native/cuda/layer_norm_kernel.cu` in v2.14),
 * so this runs it on bf16 directly.
 * 
 * @param seq The block to run.
 * @param x The tensor to pass through the block.
 * @return The result of the pass, what this block outputs.
 */
torch::Tensor run_block(torch::nn::Sequential& seq, torch::Tensor x) {
    const bool autocast_enabled = at::autocast::is_autocast_enabled(torch::kCUDA);
    for (auto& layer : *seq) {
        auto ln = std::dynamic_pointer_cast<torch::nn::LayerNormImpl>(layer.ptr());
        if (ln && autocast_enabled && x.scalar_type() == torch::kBFloat16) {
            AutocastOffGuard temp_disable_autocast_guard;
            x = torch::layer_norm(x, 
                                  ln->options.normalized_shape(), 
                                  ln->weight.to(torch::kBFloat16), 
                                  ln->bias.to(torch::kBFloat16), 
                                  ln->options.eps());
        } else {
            x = layer.forward(x);
        }
    }
    return x;
}

} // namespace

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

    // Record the Linear weights fed by raw obs. Each block's first Linear sits at index 0.
    // With no hidden layers on a path, its head reads the obs directly and is already an IO weight.
    if (shared_mlp_) {
        input_weight_names_.push_back("shared_mlp.0.weight");
    } else {
        if (actor_mlp_) input_weight_names_.push_back("actor_mlp.0.weight");
        if (critic_mlp_) input_weight_names_.push_back("critic_mlp.0.weight");
    }
}

std::vector<ParamInfo> ActorCriticImpl::parameter_layout() const {
    std::vector<ParamInfo> layout;
    for (const auto& item : named_parameters()) {
        const std::string& name = item.key();
        const std::string module = name.substr(0, name.find('.'));
        const bool is_head = module == "actor_head" || module == "critic_head";

        ParamSection section;
        if (module == "shared_mlp") {
            section = ParamSection::Shared;
        } else if (module == "actor_mlp" || module == "actor_head") {
            section = ParamSection::Actor;
        } else if (module == "critic_mlp" || module == "critic_head") {
            section = ParamSection::Critic;
        } else {
            throw std::logic_error("Unclassified ActorCritic parameter: " + name);
        }

        ParamKind kind = ParamKind::HiddenWeight;
        if (item.value().dim() == 1) {
            kind = ParamKind::Vector;
        } else if (is_head || std::find(input_weight_names_.begin(), input_weight_names_.end(), name) != input_weight_names_.end()) {
            kind = ParamKind::IOWeight;
        }

        layout.push_back({name, item.value(), section, kind});
    }
    return layout;
}

std::tuple<torch::Tensor, torch::Tensor> ActorCriticImpl::forward(torch::Tensor obs) {
    torch::Tensor shared_features = obs;
    if (shared_mlp_) {
        shared_features = run_block(shared_mlp_, shared_features);
    }

    torch::Tensor actor_features = shared_features;
    if (actor_mlp_) {
        actor_features = run_block(actor_mlp_, actor_features);
    }

    torch::Tensor critic_features = shared_features;
    if (critic_mlp_) {
        critic_features = run_block(critic_mlp_, critic_features);
    }

    torch::Tensor logits = actor_head_->forward(actor_features);
    torch::Tensor values = critic_head_->forward(critic_features);

    return { logits, values };
}

torch::Tensor ActorCriticImpl::forward_actor(torch::Tensor obs) {
    torch::Tensor shared_features = obs;
    if (shared_mlp_) {
        shared_features = run_block(shared_mlp_, shared_features);
    }
    if (actor_mlp_) {
        shared_features = run_block(actor_mlp_, shared_features);
    }
    return actor_head_->forward(shared_features);
}

torch::Tensor ActorCriticImpl::forward_critic(torch::Tensor obs) {
    torch::Tensor shared_features = obs;
    if (shared_mlp_) {
        shared_features = run_block(shared_mlp_, shared_features);
    }
    if (critic_mlp_) {
        shared_features = run_block(critic_mlp_, shared_features);
    }
    return critic_head_->forward(shared_features);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ActorCriticImpl::get_action_and_value(
    torch::Tensor obs,
    torch::Tensor action_masks
) {
    torch::NoGradGuard no_grad;

    auto [logits, values] = forward(obs);

    torch::Tensor masked_logits = mask_logits(logits, action_masks);
    torch::Tensor actions = sample_masked(masked_logits);

    torch::Tensor log_probs = torch::log_softmax(masked_logits, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1);

    return { actions, log_probs, values };
}

torch::Tensor ActorCriticImpl::sample_action(torch::Tensor obs, torch::Tensor action_masks) {
    torch::NoGradGuard no_grad;
    return sample_masked(mask_logits(forward_actor(obs), action_masks));
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