// buta-ppo/src/rl/ppo_trainer.cpp
#include "ppo_trainer.hpp"
#include "rollout_buffer.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>

namespace buta_ppo::rl {

PPOTrainer::PPOTrainer(PPOConfig config, ActorCritic actor_critic, torch::Device device)
    : config_(config), actor_critic_(actor_critic), device_(device) {
    const auto opt_opts = torch::optim::AdamOptions(config_.policy_lr);
    optimizer_ = std::make_unique<torch::optim::Adam>(actor_critic_->parameters(), opt_opts);
}

void PPOTrainer::graph_safe_clip_grad_norm(const std::vector<torch::Tensor>& parameters, float max_norm) {
    if (parameters.empty()) return;

    const auto options = parameters[0].options();
    torch::Tensor total_norm_sq = torch::zeros({1}, options);
    std::vector<torch::Tensor> grads;

    for (const auto& p : parameters) {
        if (p.grad().defined()) {
            grads.push_back(p.grad());
            total_norm_sq += p.grad().pow(2).sum();
        }
    }

    if (grads.empty()) return;

    torch::Tensor total_norm = total_norm_sq.sqrt();
    torch::Tensor clip_coef = max_norm / (total_norm + 1e-6f);
    torch::Tensor clamp_coef = torch::clamp_max(clip_coef, 1.0f);

    for (auto& g : grads) {
        g.mul_(clamp_coef);
    }
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

        // Approx KL Divergence.
        // torch::Tensor approx_kl_tsr = ((ratio - 1.0f) - log_ratio).mean();

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
        // static_approx_kl_.copy_(approx_kl_tsr.detach());
        static_value_loss_.copy_(value_loss.detach());
        static_entropy_.copy_(entropy.mean().detach());
    }

    // Optim.
    optimizer_->zero_grad();
    loss.backward();
    graph_safe_clip_grad_norm(actor_critic_->parameters(), config_.max_grad_norm);
    optimizer_->step();
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
                // static_approx_kl_ = torch::zeros({1}, float_opts);
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
                at::cuda::CUDAStreamGuard stream_guard(capture_stream);

                // Before graph warmup and creation store current model weights.
                std::vector<torch::Tensor> weight_backups;
                {
                    torch::NoGradGuard no_grad;
                    for (const auto& param : actor_critic_->parameters()) {
                        weight_backups.push_back(param.clone());
                    }
                }

                // Backup optimizer state.
                std::stringstream optimizer_backup;
                torch::save(*optimizer_, optimizer_backup);

                // Warmup for graph capture.
                // Extra optim passes can/will hurt the model.
                for (int i = 0; i < 3; i++) {
                    execute_minibatch_graph_logic();
                }

                // Restore backup.
                {
                    torch::NoGradGuard no_grad;
                    size_t idx = 0;
                    for (auto& param : actor_critic_->parameters()) {
                        param.copy_(weight_backups[idx++]);
                    }
                }

                // Restore optimizer state.
                optimizer_backup.seekg(0);
                torch::load(*optimizer_, optimizer_backup);

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

            updates++;
        }
    }

    return {
        {"policy_loss", total_policy_loss_tsr.item<float>() / std::max(1, updates)},
        {"value_loss", total_value_loss_tsr.item<float>() / std::max(1, updates)},
        {"entropy", total_entropy_tsr.item<float>() / std::max(1, updates)}
    };
}

} // namespace buta_ppo::rl