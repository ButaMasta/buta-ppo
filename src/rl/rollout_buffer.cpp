// buta-ppo/src/rl/rollout_buffer.cpp
#include "rollout_buffer.hpp"
#include <cstdint>

namespace buta_ppo::rl {

RolloutBuffer::RolloutBuffer(size_t buffer_size, size_t num_agents, size_t obs_size, size_t action_space_size, torch::Device device) 
    : buffer_size_(buffer_size), num_agents_(num_agents), obs_size_(obs_size), action_space_size_(action_space_size), device_(device) {
    
    // Allocate all tensors on the GPU.

    // Types for convenience.
    auto float32_opts = torch::TensorOptions().device(device_).dtype(torch::kFloat32);
    auto int64_opts = torch::TensorOptions().device(device_).dtype(torch::kInt64);
    auto bool_opts = torch::TensorOptions().device(device_).dtype(torch::kBool);

    obs_            = torch::zeros({(int64_t)buffer_size_, (int64_t)num_agents_, (int64_t)obs_size_}, float32_opts);

    actions_        = torch::zeros({(int64_t)buffer_size_, (int64_t)num_agents_}, int64_opts);
    action_masks_   = torch::zeros({(int64_t)buffer_size_, (int64_t)num_agents_, (int64_t)action_space_size_}, bool_opts);
    rewards_        = torch::zeros({(int64_t)buffer_size_, (int64_t)num_agents_}, float32_opts);
    dones_          = torch::zeros({(int64_t)buffer_size_, (int64_t)num_agents_}, float32_opts);
    log_probs_      = torch::zeros({(int64_t)buffer_size_, (int64_t)num_agents_}, float32_opts);
    values_         = torch::zeros({(int64_t)buffer_size_, (int64_t)num_agents_}, float32_opts);

    advantages_     = torch::zeros({(int64_t)buffer_size_, (int64_t)num_agents_}, float32_opts);
    returns_        = torch::zeros({(int64_t)buffer_size_, (int64_t)num_agents_}, float32_opts);
}

void RolloutBuffer::reset() {
    step_ = 0;
}

void RolloutBuffer::insert(
    const torch::Tensor& obs,
    const torch::Tensor& actions,
    const torch::Tensor& action_masks,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    const torch::Tensor& log_probs,
    const torch::Tensor& values
) {
    if (step_ >= buffer_size_) {
        throw std::runtime_error("RolloutBuffer is full, cannot insert more data.");
    }

    obs_[step_].copy_(obs);
    actions_[step_].copy_(actions);
    action_masks_[step_].copy_(action_masks);
    rewards_[step_].copy_(rewards);
    dones_[step_].copy_(dones);
    log_probs_[step_].copy_(log_probs);
    values_[step_].copy_(values);

    step_++;
}

void RolloutBuffer::compute_returns_and_advantages(
    const torch::Tensor& last_values,
    const torch::Tensor& last_dones,
    float gamma,
    float gae_lambda
) {
    torch::Tensor last_gae_lam = torch::zeros({(int64_t)num_agents_}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));

    int64_t max_step = static_cast<int64_t>(buffer_size_);

    for (int64_t step = max_step - 1; step >= 0; --step) {
        torch::Tensor next_non_terminal;
        torch::Tensor next_values;

        if (step == max_step - 1) {
            next_non_terminal = 1.0f - last_dones;
            next_values = last_values;
        } else {
            next_non_terminal = 1.0f - dones_[step + 1];
            next_values = values_[step + 1];
        }

        // TD Error (delta): r + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        torch::Tensor delta = rewards_[step] + gamma * next_values * next_non_terminal - values_[step];

        // GAE: delta + gamma * lambda * (1 - done) * last_gae
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam;

        advantages_[step].copy_(last_gae_lam);
    }

    // Advantages + Critic's prediction.
    returns_ = advantages_ + values_;
}

}; // namespace buta_ppo::rl