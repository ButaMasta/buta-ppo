// buta-ppo/src/rl/ppo_runner.hpp
#include "buta_ppo/ffi.h"
#include "ppo_trainer.hpp"
#include "rollout_buffer.hpp"
#include "env/vec_env.hpp"
#include <cstddef>
#include <stdexcept>
#include <torch/torch.h>
#include <memory>
#include <atomic>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>


namespace buta_ppo::rl {

struct MatchDistribution {
    size_t blue_players = 1;
    size_t orange_players = 1;
    float weight = 1.0f;

    [[nodiscard]] size_t total_players() const {
        return blue_players + orange_players;
    }

    [[nodiscard]] std::vector<ffi::Team> to_team_layout() const {
        std::vector<ffi::Team> layout;
        layout.reserve(total_players());
        layout.insert(layout.end(), blue_players, ffi::Team::Blue);
        layout.insert(layout.end(), orange_players, ffi::Team::Orange);
        return layout;
    }
};

struct RunnerConfig {
    std::string bot_name = "default";
    bool render = false;

    std::vector<MatchDistribution> match_distributions = {
        {1, 1, 1.0f}
    };
    size_t num_envs = 256;
    size_t num_threads = 12;
    size_t max_players_per_team = 3;
    int ticks_per_step = 8;

    size_t target_steps_per_update = 50'000;
    size_t num_minibatches = 4;

    ActorCriticConfig ac_cfg;
    PPOConfig ppo_cfg;
};

inline std::vector<size_t> compute_env_counts(
    size_t target_total_envs,
    const std::vector<MatchDistribution>& distributions
) {
    if (distributions.empty()) {
        throw std::invalid_argument("Match distributions cannot be empty.");
    }

    float weight_sum = 0.0f;
    for (const auto& dist : distributions) {
        if (dist.weight < 0.0f) {
            throw std::invalid_argument("Distribution weights must be non-negative.");
        }
        weight_sum += dist.weight;
    }

    if (weight_sum <= 0.0f) {
        throw std::invalid_argument("Sum of distributions must be positive.");
    }

    std::vector<size_t> counts(distributions.size(), 0);
    size_t allocated = 0;
    size_t max_weight_idx = 0;
    float max_weight = -1.0f;

    for (size_t i = 0; i < distributions.size(); i++) {
        float norm_weight = distributions[i].weight / weight_sum;
        counts[i] = static_cast<size_t>(target_total_envs * norm_weight);
        allocated += counts[i];

        if (distributions[i].weight > max_weight) {
            max_weight = distributions[i].weight;
            max_weight_idx = i;
        }
    }

    // Allocate remainder to highest weight. Not exact but it should apply to the vast majority of cases.
    size_t remainder = target_total_envs - allocated;
    counts[max_weight_idx] += remainder;

    return counts;
}

class PPORunner {
private:
    RunnerConfig config_;
    torch::Device device_;

    std::unique_ptr<env::VecEnv> vec_env_;
    ActorCritic actor_critic_{nullptr};
    std::unique_ptr<PPOTrainer> trainer_;
    std::unique_ptr<RolloutBuffer> buffer_;

    size_t total_agents_;
    size_t buffer_size_;
    int64_t total_steps_per_update_;

    torch::Tensor actions_cpu_;
    torch::Tensor step_obs_gpu_;
    torch::Tensor step_masks_gpu_;
    torch::Tensor step_rewards_gpu_;
    torch::Tensor step_dones_gpu_;

    int64_t global_step_{0};

    void setup_dimensions_and_buffers();
    int64_t load_latest_checkpoint(const std::string& dir);
    
    void run_training(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir);
    void run_render(const std::atomic<bool>& stop_flag);

public:
    explicit PPORunner(const RunnerConfig& config);

    void save_checkpoint(const std::string& dir) const;
    void run(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir);
};

}; // namespace buta_ppo::rl