// buta-ppo/src/rl/ppo_runner.hpp
#include "ppo_trainer.hpp"
#include "rollout_buffer.hpp"
#include "env/vec_env.hpp"
#include <torch/torch.h>
#include <memory>
#include <atomic>
#include <cstring>


namespace buta_ppo::rl {

struct RunnerConfig {
    size_t num_envs = 256;
    size_t agents_per_env = 2;
    int ticks_per_step = 8;

    size_t target_steps_per_update = 50'000;
    size_t num_minibatches = 4;

    ActorCriticConfig ac_cfg;
    PPOConfig ppo_cfg;
};

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

    void setup_dimensions_and_buffers();

public:
    explicit PPORunner(const RunnerConfig& config);

    void save_checkpoint(const std::string& path) const;

    void run(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir);
};

}; // namespace buta_ppo::rl