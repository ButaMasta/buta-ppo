// buta-ppo/src/rl/ppo_trainer.cpp
#include "ppo_trainer.hpp"
#include "rollout_buffer.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

#include <algorithm>
#include <iostream>
#include <utility>

namespace buta_ppo::rl {

PPOTrainer::PPOTrainer(PPOConfig config, ActorCritic actor_critic, torch::Device device)
    : config_(config), actor_critic_(actor_critic), device_(device) {
    const std::vector<ParamInfo> layout = actor_critic_->parameter_layout();
    for (const auto& info : layout) {
        param_names_.push_back(info.name);
        params_.push_back(info.tensor);
    }
    grads_.reserve(params_.size());

    int64_t shared_count = 0, actor_count = 0, critic_count = 0;
    for (const auto& info : layout) {
        switch (info.section) {
            case ParamSection::Shared: shared_count += info.tensor.numel(); break;
            case ParamSection::Actor:  actor_count += info.tensor.numel(); break;
            case ParamSection::Critic: critic_count += info.tensor.numel(); break;
        }
    }
    std::cout << "Model Parameters: " << shared_count + actor_count + critic_count
              << "\n | Shared: " << shared_count
              << "\n | Actor:  " << actor_count
              << "\n | Critic: " << critic_count << "\n";

    optimizers_ = build_optimizers(layout, config_.optimizer);
    // std::cout << "Optimizers:\n";
    // for (const auto& opt : optimizers_) {
    //     std::cout << " | " << opt->describe() << "\n";
    // }
}

std::vector<std::pair<std::string, torch::Tensor>> PPOTrainer::optimizer_state() const {
    std::vector<std::pair<std::string, torch::Tensor>> state;
    for (const auto& opt : optimizers_) {
        auto opt_state = opt->named_state();
        state.insert(state.end(), std::make_move_iterator(opt_state.begin()), std::make_move_iterator(opt_state.end()));
    }
    return state;
}

void PPOTrainer::save_optimizer(const std::string& path) const {
    torch::serialize::OutputArchive archive;
    for (const auto& [key, tensor] : optimizer_state()) {
        archive.write(key, tensor, /*is_buffer=*/true);
    }
    archive.save_to(path);
}

bool PPOTrainer::load_optimizer(const std::string& path) {
    torch::serialize::InputArchive archive;
    try {
        archive.load_from(path, device_);
    } catch (const std::exception& e) {
        std::cerr << "[WARN] Could not read optimizer state from " << path << ": " << e.what() << "\n";
        return false;
    }

    // Read and validate every entry before touching the live state so a mismatch leaves it untouched.
    // A changed network or optimizer type shows up here as a missing key or a shape mismatch.
    const auto state = optimizer_state();
    std::vector<torch::Tensor> loaded;
    loaded.reserve(state.size());
    for (const auto& [key, target] : state) {
        torch::Tensor t;
        if (!archive.try_read(key, t, /*is_buffer=*/true) || !t.sizes().equals(target.sizes())) {
            std::cerr << "[WARN] Optimizer state in " << path << " does not match the current network and optimizers at '"
                      << key << "'.\n";
            return false;
        }
        loaded.push_back(t);
    }

    // Copy into the existing tensors so their device addresses (used by the CUDA graph) stay fixed.
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < state.size(); ++i) {
        state[i].second.copy_(loaded[i]);
    }
    return true;
}

void PPOTrainer::graph_safe_clip_grad_norm(const std::vector<torch::Tensor>& grads, float max_norm) {
    if (grads.empty()) return;

    torch::NoGradGuard no_grad;

    // Multi-tensor kernels: per-grad norms in one launch, then the global norm from the stacked norms.
    const std::vector<torch::Tensor> norms = torch::_foreach_norm(grads, 2);
    torch::Tensor total_norm = torch::linalg_vector_norm(torch::stack(norms), 2);
    torch::Tensor clip_coef = torch::clamp_max(max_norm / (total_norm + 1e-6f), 1.0f);

    torch::_foreach_mul_(grads, clip_coef);
}

void PPOTrainer::execute_minibatch_graph_logic() {
    torch::Tensor loss;
    {
        BFloat16AutocastGuard amp_guard;

        auto [new_log_probs, entropy, new_values] = actor_critic_->evaluate_actions(
            static_mb_obs_, static_mb_actions_, static_mb_action_masks_);

        // Policy Loss
        torch::Tensor log_ratio = new_log_probs - static_mb_old_log_probs_;
        torch::Tensor ratio = log_ratio.exp();

        // Policy-change diagnostics: how far the current policy has moved from the rollout policy on this minibatch.
        // Approx KL uses the low-variance (ratio - 1) - log(ratio) estimator, which is always >= 0.
        torch::Tensor approx_kl = ((ratio - 1.0f) - log_ratio).mean();
        torch::Tensor clip_fraction = ((ratio - 1.0f).abs() > config_.clip_ratio).to(torch::kFloat32).mean();

        torch::Tensor pg_loss1 = static_mb_advantages_ * ratio;
        torch::Tensor pg_loss2 = static_mb_advantages_ * torch::clamp(ratio, 1.0f - config_.clip_ratio, 1.0f + config_.clip_ratio);
        torch::Tensor policy_loss = -torch::min(pg_loss1, pg_loss2).mean();

        // Value Loss.
        torch::Tensor value_loss = torch::nn::functional::mse_loss(new_values.squeeze(-1), static_mb_returns_);

        // Entropy Bonus.
        torch::Tensor entropy_loss = -entropy.mean() * config_.entropy_coef;

        // Total Loss.
        loss = policy_loss + config_.value_coef * value_loss + entropy_loss;

        // Accumulate metrics.
        static_policy_loss_.copy_(policy_loss.detach());
        static_approx_kl_.copy_(approx_kl.detach());
        static_clip_fraction_.copy_(clip_fraction.detach());
        static_value_loss_.copy_(value_loss.detach());
        static_entropy_.copy_(entropy.mean().detach());
    }

    // Optim.
    for (auto& p : params_) {
        p.mutable_grad().reset(); // zero_grad(set_to_none)
    }
    loss.backward();

    // Gather grad handles once for clipping and the optimizer. Refills reserved capacity, no heap allocation.
    grads_.clear();
    for (const auto& p : params_) {
        TORCH_CHECK(p.grad().defined(), "Every parameter must receive a gradient for the graph-safe optimizers.");
        grads_.push_back(p.grad());
    }

    graph_safe_clip_grad_norm(grads_, config_.max_grad_norm);
    for (const auto& opt : optimizers_) {
        opt->step();
    }
}

std::unordered_map<std::string, float> PPOTrainer::train_step(const RolloutBuffer& buffer) {
    actor_critic_->train();

    // Flatten buffers.
    const int64_t total_batch_size = buffer.obs_.size(0) * buffer.obs_.size(1);

    torch::Tensor b_obs = buffer.obs_.view({ total_batch_size, -1 });
    torch::Tensor b_actions = buffer.actions_.view({ total_batch_size });
    torch::Tensor b_action_masks = buffer.action_masks_.view({ total_batch_size, -1 });
    torch::Tensor b_log_probs = buffer.log_probs_.view({ total_batch_size });
    torch::Tensor b_advantages = buffer.advantages_.view({ total_batch_size });
    torch::Tensor b_returns = buffer.returns_.view({ total_batch_size });

    // Norm advantages.
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8f);

    // Prepare index buffer.
    if (!batch_indices_.defined() || batch_indices_.size(0) != total_batch_size) {
        batch_indices_ = torch::empty({ total_batch_size }, torch::TensorOptions().device(device_).dtype(torch::kInt64));
    }

    // Prepare metrics.
    torch::Tensor total_policy_loss_tsr = torch::zeros({1}, torch::TensorOptions().device(device_));
    torch::Tensor total_value_loss_tsr = torch::zeros({1}, torch::TensorOptions().device(device_));
    torch::Tensor total_entropy_tsr = torch::zeros({1}, torch::TensorOptions().device(device_));
    torch::Tensor total_approx_kl_tsr = torch::zeros({1}, torch::TensorOptions().device(device_));
    torch::Tensor total_clip_fraction_tsr = torch::zeros({1}, torch::TensorOptions().device(device_));
    
    int updates = 0;
    // bool early_stop = false; // Ommited, was only used when KL divergence was monitored.

    // PPO Epoch loop.
    for (int epoch = 0; epoch < config_.epochs; ++epoch) {
        // if (early_stop) break;

        // Shuffle.
        torch::randperm_out(batch_indices_, total_batch_size);

        for (int64_t start = 0; start < total_batch_size; start += config_.mini_batch_size) {
            const int64_t end = std::min(start + config_.mini_batch_size, total_batch_size);
            const torch::Tensor mb_inds = batch_indices_.slice(0, start, end);

            if (!static_mb_obs_.defined()) {
                const auto float_opts = torch::TensorOptions().device(device_).dtype(torch::kFloat32);
                const auto int_opts = torch::TensorOptions().device(device_).dtype(torch::kInt64);
                const auto bool_opts = torch::TensorOptions().device(device_).dtype(torch::kBool);

                const int64_t mb_size = mb_inds.size(0);

                static_mb_obs_ = torch::empty({mb_size, b_obs.size(1)}, float_opts);
                static_mb_actions_ = torch::empty({mb_size}, int_opts);
                static_mb_action_masks_ = torch::empty({mb_size, b_action_masks.size(1)}, bool_opts);
                static_mb_old_log_probs_ = torch::empty({mb_size}, float_opts);
                static_mb_advantages_ = torch::empty({mb_size}, float_opts);
                static_mb_returns_ = torch::empty({mb_size}, float_opts);

                static_policy_loss_ = torch::zeros({1}, float_opts);
                static_approx_kl_ = torch::zeros({1}, float_opts);
                static_clip_fraction_ = torch::zeros({1}, float_opts);
                static_value_loss_ = torch::zeros({1}, float_opts);
                static_entropy_ = torch::zeros({1}, float_opts);
            }

            torch::index_select_out(static_mb_obs_, b_obs, 0, mb_inds);
            torch::index_select_out(static_mb_actions_, b_actions, 0, mb_inds);
            torch::index_select_out(static_mb_action_masks_, b_action_masks, 0, mb_inds);
            torch::index_select_out(static_mb_old_log_probs_, b_log_probs, 0, mb_inds);
            torch::index_select_out(static_mb_advantages_, b_advantages, 0, mb_inds);
            torch::index_select_out(static_mb_returns_, b_returns, 0, mb_inds);

            if (!graph_captured_) {

                // Graph capture has to happen on a private stream.
                at::cuda::CUDAStream capture_stream = at::cuda::getStreamFromPool();

                // The capture stream must wait for the minibatch gathers queued on the current stream.
                {
                    at::cuda::CUDAEvent inputs_ready;
                    inputs_ready.record(at::cuda::getCurrentCUDAStream());
                    inputs_ready.block(capture_stream);
                }
                at::cuda::CUDAStreamGuard stream_guard(capture_stream);

                // Back up all training state (weights + optimizer state) so warmup passes leave no trace.
                std::vector<torch::Tensor> train_state = params_;
                for (const auto& [key, tensor] : optimizer_state()) {
                    train_state.push_back(tensor);
                }
                std::vector<torch::Tensor> backups;
                {
                    torch::NoGradGuard no_grad;
                    for (const auto& t : train_state) {
                        backups.push_back(t.clone());
                    }
                }

                // Warmup for graph capture.
                // Extra optim passes can/will hurt the model.
                for (int i = 0; i < 3; i++) {
                    execute_minibatch_graph_logic();
                }

                // Restore backup.
                {
                    torch::NoGradGuard no_grad;
                    for (size_t i = 0; i < train_state.size(); ++i) {
                        train_state[i].copy_(backups[i]);
                    }
                }

                capture_stream.synchronize();

                // Capture the graph.
                graph_.capture_begin();
                execute_minibatch_graph_logic();
                graph_.capture_end();

                graph_captured_ = true;
            } else {
                graph_.replay();
            }

            total_policy_loss_tsr += static_policy_loss_;
            total_value_loss_tsr += static_value_loss_;
            total_entropy_tsr += static_entropy_;
            total_approx_kl_tsr += static_approx_kl_;
            total_clip_fraction_tsr += static_clip_fraction_;

            updates++;
        }
    }

    return {
        {"policy_loss", total_policy_loss_tsr.item<float>() / std::max(1, updates)},
        {"value_loss", total_value_loss_tsr.item<float>() / std::max(1, updates)},
        {"entropy", total_entropy_tsr.item<float>() / std::max(1, updates)},
        {"approx_kl", total_approx_kl_tsr.item<float>() / std::max(1, updates)},
        {"clip_fraction", total_clip_fraction_tsr.item<float>() / std::max(1, updates)}
    };
}

} // namespace buta_ppo::rl