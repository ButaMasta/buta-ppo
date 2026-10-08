// buta-ppo/src/rl/rollout_buffer.cpp
#include "rollout_buffer.hpp"
#include <cstdint>
#include <stdexcept>

namespace buta_ppo::rl {

RolloutBuffer::RolloutBuffer(size_t buffer_size, size_t num_agents, size_t obs_size, size_t action_space_size, torch::Device device) 
    : buffer_size_(buffer_size), num_agents_(num_agents), obs_size_(obs_size), action_space_size_(action_space_size), device_(device) {
    
    // Allocate all tensors on the GPU.

    // Types for convenience.
    const auto float32_opts = torch::TensorOptions().device(device_).dtype(torch::kFloat32);
    const auto int64_opts = torch::TensorOptions().device(device_).dtype(torch::kInt64);
    const auto bool_opts = torch::TensorOptions().device(device_).dtype(torch::kBool);

    const int64_t b_size = static_cast<int64_t>(buffer_size_);
    const int64_t n_agents = static_cast<int64_t>(num_agents_);
    const int64_t o_size = static_cast<int64_t>(obs_size_);
    const int64_t a_size = static_cast<int64_t>(action_space_size_);

    obs_            = torch::zeros({b_size, n_agents, o_size}, float32_opts);
    actions_        = torch::zeros({b_size, n_agents}, int64_opts);
    action_masks_   = torch::zeros({b_size, n_agents, a_size}, bool_opts);
    rewards_        = torch::zeros({b_size, n_agents}, float32_opts);
    dones_          = torch::zeros({b_size, n_agents}, float32_opts);
    log_probs_      = torch::zeros({b_size, n_agents}, float32_opts);
    values_         = torch::zeros({b_size, n_agents}, float32_opts);
    advantages_     = torch::zeros({b_size, n_agents}, float32_opts);
    returns_        = torch::zeros({b_size, n_agents}, float32_opts);

    // Allocate CPU tensors.
    const auto pinned_opts = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32).pinned_memory(true);
    cpu_rewards_    = torch::zeros({b_size, n_agents}, pinned_opts);
    cpu_values_     = torch::zeros({b_size, n_agents}, pinned_opts);
    cpu_dones_      = torch::zeros({b_size, n_agents}, pinned_opts);
    cpu_advantages_ = torch::zeros({b_size, n_agents}, pinned_opts);
    cpu_last_val_   = torch::zeros({n_agents}, pinned_opts);

    last_gae_.resize(num_agents_, 0.0f);
}

void RolloutBuffer::insert(
    size_t step,
    int64_t agent_offset,
    const torch::Tensor& obs,
    const torch::Tensor& actions,
    const torch::Tensor& action_masks,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    const torch::Tensor& log_probs,
    const torch::Tensor& values
) {
    const int64_t count = obs.size(0);
    if (step >= buffer_size_ || agent_offset < 0 || agent_offset + count > static_cast<int64_t>(num_agents_)) {
        throw std::out_of_range("RolloutBuffer insert is outside the buffer.");
    }

    obs_[step].narrow(0, agent_offset, count).copy_(obs);
    actions_[step].narrow(0, agent_offset, count).copy_(actions);
    action_masks_[step].narrow(0, agent_offset, count).copy_(action_masks);
    rewards_[step].narrow(0, agent_offset, count).copy_(rewards);
    dones_[step].narrow(0, agent_offset, count).copy_(dones);
    log_probs_[step].narrow(0, agent_offset, count).copy_(log_probs);
    values_[step].narrow(0, agent_offset, count).copy_(values);
}

void RolloutBuffer::compute_returns_and_advantages(
    const torch::Tensor& last_values,
    float gamma,
    float gae_lambda
) {
    // Transfer to CPU for GAE.
    cpu_rewards_.copy_(rewards_, true);
    cpu_values_.copy_(values_, true);
    cpu_dones_.copy_(dones_, true);
    cpu_last_val_.copy_(last_values, true);

    // Ensure all copies are done before accessing pointers.
    torch::cuda::synchronize();
    
    const float* r_ptr = cpu_rewards_.data_ptr<float>();
    const float* v_ptr = cpu_values_.data_ptr<float>();
    const float* d_ptr = cpu_dones_.data_ptr<float>();
    float* adv_ptr = cpu_advantages_.data_ptr<float>();
    const float* lv_ptr = cpu_last_val_.data_ptr<float>();
    
    const int64_t n_steps = buffer_size_;
    const int64_t n_agents = num_agents_;

    std::fill(last_gae_.begin(), last_gae_.end(), 0.0f);
    
    // Compute GAE.
    for (int64_t step = n_steps - 1; step >= 0; --step) {
        for (int64_t agent = 0; agent < n_agents; ++agent) {
            const int64_t idx = step * n_agents + agent;
            
            const float next_val = (step == n_steps - 1) ? lv_ptr[agent] : v_ptr[idx + n_agents];
            // dones_[step] marks whether the action at `step` ended the episode, so it masks this step's own bootstrap.
            const float next_non_term = 1.0f - d_ptr[idx];
            
            const float delta = r_ptr[idx] + gamma * next_val * next_non_term - v_ptr[idx];
            last_gae_[agent] = delta + gamma * gae_lambda * next_non_term * last_gae_[agent];
            adv_ptr[idx] = last_gae_[agent];
        }
    }
    
    // Transfer GAE back to GPU.
    advantages_.copy_(cpu_advantages_, true);
    returns_ = advantages_ + values_;
}

} // namespace buta_ppo::rl