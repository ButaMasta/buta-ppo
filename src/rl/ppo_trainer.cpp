// buta-ppo/src/rl/ppo_trainer.cpp
#include "ppo_trainer.hpp"

namespace buta_ppo::rl {

PPOTrainer::PPOTrainer(PPOConfig config, ActorCritic actor_critic, torch::Device device)
    : config_(config), actor_critic_(actor_critic), device_(device) {
    auto opt_opts = torch::optim::AdamOptions(config_.policy_lr);
    optimizer_ = std::make_unique<torch::optim::Adam>(actor_critic_->parameters(), opt_opts);
}

std::unordered_map<std::string, float> PPOTrainer::train_step(const RolloutBuffer& buffer) {
    actor_critic_->train();

    // Flatten buffers.
    int64_t total_batch_size = buffer.obs_.size(0) * buffer.obs_.size(1);

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
    float total_policy_loss = 0.0f;
    float total_value_loss = 0.0f;
    float total_entropy = 0.0f;
    int updates = 0;
    bool early_stop = false;

    // PPO Epoch loop.
    for (int epoch = 0; epoch < config_.epochs; ++epoch) {
        if (early_stop) break;

        // Shuffle.
        torch::randperm_out(batch_indices_, total_batch_size);

        for (int64_t start = 0; start < total_batch_size; start += config_.mini_batch_size) {
            int64_t end = std::min(start + config_.mini_batch_size, total_batch_size);
            torch::Tensor mb_inds = batch_indices_.slice(0, start, end);

            torch::Tensor mb_obs = b_obs.index_select(0, mb_inds);
            torch::Tensor mb_actions = b_actions.index_select(0, mb_inds);
            torch::Tensor mb_action_masks = b_action_masks.index_select(0, mb_inds);
            torch::Tensor mb_old_log_probs = b_log_probs.index_select(0, mb_inds);
            torch::Tensor mb_advantages = b_advantages.index_select(0, mb_inds);
            torch::Tensor mb_returns = b_returns.index_select(0, mb_inds);

            auto [new_log_probs, entropy, new_values] = actor_critic_->evaluate_actions(mb_obs, mb_actions, mb_action_masks);

            // Policy Loss
            torch::Tensor log_ratio = new_log_probs - mb_old_log_probs;
            torch::Tensor ratio = log_ratio.exp();

            // Approx KL Divergence.
            float approx_kl = ((ratio - 1.0f) - log_ratio).mean().item<float>();
            if (approx_kl > config_.target_kl * 1.5f) {
                early_stop = true;
                break;
            }

            torch::Tensor pg_loss1 = mb_advantages * ratio;
            torch::Tensor pg_loss2 = mb_advantages * torch::clamp(ratio, 1.0f - config_.clip_ratio, 1.0f + config_.clip_ratio);
            torch::Tensor policy_loss = -torch::min(pg_loss1, pg_loss2).mean();

            // Value Loss.
            torch::Tensor value_loss = torch::nn::functional::mse_loss(new_values.squeeze(-1), mb_returns);

            // Entropy Bonus.
            torch::Tensor entropy_loss = -entropy.mean() * config_.entropy_coef;

            // Total Loss.
            torch::Tensor loss = policy_loss + config_.value_coef * value_loss + entropy_loss;

            // Optim.
            optimizer_->zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(actor_critic_->parameters(), config_.max_grad_norm);
            optimizer_->step();

            // Metrics.
            total_policy_loss += policy_loss.item<float>();
            total_value_loss += value_loss.item<float>();
            total_entropy += entropy.mean().item<float>();
            updates++;
        }
    }

    return {
        {"policy_loss", total_policy_loss / std::max(1, updates)},
        {"value_loss", total_value_loss / std::max(1, updates)},
        {"entropy", total_entropy / std::max(1, updates)}
    };
}

}; // namespace buta_ppo::rl