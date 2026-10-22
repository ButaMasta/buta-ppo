// buta-ppo/src/rl/actor_critic.cpp
#include "actor_critic.hpp"

#include <ATen/autocast_mode.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace buta_ppo::rl {

namespace {

// The sub-archive of a model file that holds its `ModelConfig`, next to the network's weights.
constexpr const char* CONFIG_KEY = "buta_config";
// Raised when the stored fields change, so an older build refuses a config it can't read instead of misreading it.
constexpr int64_t CONFIG_FORMAT_VERSION = 1;

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

/**
 * @brief Pairs each of a module's tensors with the archive's, recursing into child modules like `Module::load`.
 *
 * Throws on a missing key, a shape mismatch, or a key in the archive the module doesn't have.
 *
 * @param module The module to read into.
 * @param archive The archive `module`'s weights were saved to.
 * @param prefix The module's dotted path, for error messages.
 * @param pairs Gets each (module tensor, archive tensor) pair.
 */
void match_archive(torch::nn::Module& module, torch::serialize::InputArchive& archive, const std::string& prefix,
                   std::vector<std::pair<torch::Tensor, torch::Tensor>>& pairs) {
    std::vector<std::string> expected;

    const auto match_tensor = [&](const std::string& name, const torch::Tensor& live) {
        torch::Tensor saved;
        if (!archive.try_read(name, saved)) {
            throw std::runtime_error("the file has no '" + prefix + name + "'");
        }
        if (!saved.sizes().equals(live.sizes())) {
            throw std::runtime_error("'" + prefix + name + "' has shape " + c10::str(saved.sizes()) +
                                     " in the file but " + c10::str(live.sizes()) + " in the network");
        }
        pairs.emplace_back(live, saved);
        expected.push_back(name);
    };

    for (const auto& item : module.named_parameters(/*recurse=*/false)) {
        match_tensor(item.key(), item.value());
    }
    for (const auto& item : module.named_buffers(/*recurse=*/false)) {
        match_tensor(item.key(), item.value());
    }
    for (const auto& child : module.named_children()) {
        if (!child.value()->is_serializable()) continue;
        torch::serialize::InputArchive child_archive;
        if (!archive.try_read(child.key(), child_archive)) {
            throw std::runtime_error("the file has no layer '" + prefix + child.key() + "'");
        }
        match_archive(*child.value(), child_archive, prefix + child.key() + ".", pairs);
        expected.push_back(child.key());
    }

    // The same shapes for every tensor here doesn't rule out the file having more, e.g. an extra final layer.
    for (const std::string& key : archive.keys()) {
        if (prefix.empty() && key == CONFIG_KEY) continue;
        if (std::find(expected.begin(), expected.end(), key) == expected.end()) {
            throw std::runtime_error("the file has '" + prefix + key + "', which the network doesn't");
        }
    }
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

void ActorCriticImpl::load_weights(const std::string& path) {
    // Read straight onto the network's device, so a checkpoint saved on the GPU also loads on a CPU-only machine.
    const torch::Device device = parameters().front().device();
    torch::serialize::InputArchive archive;
    try {
        archive.load_from(path, device);
    } catch (const c10::Error& e) {
        throw std::runtime_error("could not read " + path + ": " + e.what_without_backtrace());
    }

    // Check everything before copying anything, so a mismatch leaves the network untouched.
    std::vector<std::pair<torch::Tensor, torch::Tensor>> pairs;
    try {
        match_archive(*this, archive, "", pairs);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(path + " doesn't match the network: " + e.what());
    }

    torch::NoGradGuard no_grad;
    for (auto& [live, saved] : pairs) {
        live.copy_(saved);
    }
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

torch::Tensor ActorCriticImpl::best_action(torch::Tensor obs, torch::Tensor action_masks) {
    torch::NoGradGuard no_grad;
    return mask_logits(forward_actor(obs), action_masks).argmax(-1);
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

void save_model(const ActorCritic& model, const ModelConfig& config, const std::string& path) {
    // What `torch::save` writes, plus the config as a sub-archive.
    torch::serialize::OutputArchive archive;
    model->save(archive);

    torch::serialize::OutputArchive config_archive(archive.compilation_unit());
    config_archive.write("format_version", c10::IValue(CONFIG_FORMAT_VERSION));
    config_archive.write("obs_size", c10::IValue(config.network.obs_size));
    config_archive.write("action_size", c10::IValue(config.network.action_size));
    config_archive.write("shared_layers", c10::IValue(config.network.shared_layers));
    config_archive.write("actor_layers", c10::IValue(config.network.actor_layers));
    config_archive.write("critic_layers", c10::IValue(config.network.critic_layers));
    config_archive.write("use_layer_norm", c10::IValue(config.network.use_layer_norm));
    config_archive.write("max_players_per_team", c10::IValue(config.max_players_per_team));
    config_archive.write("ticks_per_step", c10::IValue(config.ticks_per_step));
    archive.write(CONFIG_KEY, config_archive);

    archive.save_to(path);
}

std::optional<ModelConfig> read_model_config(const std::string& path) {
    torch::serialize::InputArchive archive;
    try {
        archive.load_from(path, torch::kCPU);
    } catch (const c10::Error& e) {
        throw std::runtime_error("could not read " + path + ": " + e.what_without_backtrace());
    }

    torch::serialize::InputArchive config_archive;
    if (!archive.try_read(CONFIG_KEY, config_archive)) {
        return std::nullopt;
    }

    const auto read = [&](const std::string& key) {
        c10::IValue value;
        if (!config_archive.try_read(key, value)) {
            throw std::runtime_error(path + " has an incomplete model config, missing '" + key + "'");
        }
        return value;
    };

    const int64_t version = read("format_version").toInt();
    if (version > CONFIG_FORMAT_VERSION) {
        throw std::runtime_error(path + " was saved by a newer version of buta-ppo (model config version " +
                                 std::to_string(version) + ", this build reads up to " +
                                 std::to_string(CONFIG_FORMAT_VERSION) + ")");
    }

    ModelConfig config;
    config.network.obs_size = read("obs_size").toInt();
    config.network.action_size = read("action_size").toInt();
    config.network.shared_layers = read("shared_layers").toIntVector();
    config.network.actor_layers = read("actor_layers").toIntVector();
    config.network.critic_layers = read("critic_layers").toIntVector();
    config.network.use_layer_norm = read("use_layer_norm").toBool();
    config.max_players_per_team = read("max_players_per_team").toInt();
    config.ticks_per_step = read("ticks_per_step").toInt();
    return config;
}

std::string describe_model_config(const ModelConfig& config) {
    const auto sizes = [](const std::vector<int64_t>& layers) {
        std::string out = "{";
        for (size_t i = 0; i < layers.size(); i++) {
            out += (i > 0 ? ", " : "") + std::to_string(layers[i]);
        }
        return out + "}";
    };
    return "obs_size " + std::to_string(config.network.obs_size) +
           ", action_size " + std::to_string(config.network.action_size) +
           ", shared_layers " + sizes(config.network.shared_layers) +
           ", actor_layers " + sizes(config.network.actor_layers) +
           ", critic_layers " + sizes(config.network.critic_layers) +
           ", use_layer_norm " + (config.network.use_layer_norm ? "true" : "false") +
           ", max_players_per_team " + std::to_string(config.max_players_per_team) +
           ", ticks_per_step " + std::to_string(config.ticks_per_step);
}

} // namespace buta_ppo::rl 