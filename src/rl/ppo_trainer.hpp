// buta-ppo/src/rl/ppo_trainer.hpp
#pragma once

#include "actor_critic.hpp"
#include "rollout_buffer.hpp"
#include <torch/torch.h>

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

    float policy_lr = 3e-4f;
    float critic_lr = 3e-4f;
};

/**
 * @brief Core training class that encapsulates the model and updating it according to collected experience.
 */
class PPOTrainer {
private:
    PPOConfig config_;
    ActorCritic actor_critic_;
    std::unique_ptr<torch::optim::Adam> optimizer_;
    torch::Device device_;

    // Mini-batch shuffling.
    torch::Tensor batch_indices_;

public:
    PPOTrainer(PPOConfig config, ActorCritic actor_critic, torch::Device device);

    /**
     * @brief Executes PPO optimization loop over collected rollouts.
     * 
     * @param buffer - The Rollout buffer containing all the data for training.
     * @return std::unordered_map<std::string, float> - Metrics.
     */
    std::unordered_map<std::string, float> train_step(const RolloutBuffer& buffer);
};

}; // namespace buta_ppo::rl