// buta-ppo/src/rl/rollout_buffer.hpp
#pragma once

#include <torch/torch.h>

namespace buta_ppo::rl {

class RolloutBuffer {
private:
    size_t buffer_size_;
    size_t num_agents_;
    size_t obs_size_;
    size_t action_space_size_;
    torch::Device device_;

    size_t step_{0};

public:
    // Primary tensor storage.
    torch::Tensor obs_;
    torch::Tensor actions_;
    torch::Tensor action_masks_;
    torch::Tensor rewards_;
    torch::Tensor dones_;
    torch::Tensor log_probs_;
    torch::Tensor values_;

    // Computed tensors.
    torch::Tensor advantages_;
    torch::Tensor returns_;

    RolloutBuffer(size_t buffer_size, size_t num_agents, size_t obs_size, size_t action_space_size, torch::Device device);

    // Resets internal step counter for a new rollout.
    void reset();

    // Inserts a single step of data into the buffer at the current step index.
    void insert(
        const torch::Tensor& obs,
        const torch::Tensor& actions,
        const torch::Tensor& action_masks,
        const torch::Tensor& rewards,
        const torch::Tensor& dones,
        const torch::Tensor& log_probs,
        const torch::Tensor& values
    );

    // Calculates the GAE and Returns after the rollout phase.
    void compute_returns_and_advantages(
        const torch::Tensor& last_values,
        const torch::Tensor& last_dones,
        float gamma = 0.99f,
        float gae_lambda = 0.95f
    );

    [[nodiscard]] bool is_full() const { return step_ >= buffer_size_; };
    [[nodiscard]] size_t get_buffer_size() const { return buffer_size_; };
};


}; // namespace buta_ppo::rl