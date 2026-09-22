// buta-ppo/src/rl/ppo_runner.hpp
#include "buta_ppo/ffi.h"
#include "ppo_trainer.hpp"
#include "rollout_buffer.hpp"
#include "env/vec_env.hpp"
#include "tensorboard_logger.h"
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

/**
 * @brief Formatter for match distributions.
 * 
 * Allows the user to neatly define match disctibutions along 
 * with their associated weight to be represented within the envs.
 */
struct MatchDistribution {
    size_t blue_players = 1;
    size_t orange_players = 1;
    float weight = 1.0f;

    /**
     * @brief Total players in this match disctribution.
     * 
     * @return size_t - total.
     */
    [[nodiscard]] size_t total_players() const {
        return blue_players + orange_players;
    }

    /**
     * @brief Translates from this struct format to a team layout using the FFI Team enum.
     * 
     * @return std::vector<ffi::Team> - The vector of all team member's Team enum.
     */
    [[nodiscard]] std::vector<ffi::Team> to_team_layout() const {
        std::vector<ffi::Team> layout;
        layout.reserve(total_players());
        layout.insert(layout.end(), blue_players, ffi::Team::Blue);
        layout.insert(layout.end(), orange_players, ffi::Team::Orange);
        return layout;
    }
};

/**
 * @brief The core config for a bot trained with this framework.
 * 
 * Contains all of the configuration values for a bot with defaults in place 
 * allowing the user to have a baseline before customizing it to their needs.
 */
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

/**
 * @brief Given the desired match distributions, evenly allocate environments to represent them.
 * 
 * @param target_total_envs - The total envs to have.
 * @param distributions - The distributions and weights to allocate envs to.
 * @return std::vector<size_t> - The number of envs correlating to the match distributions.
 */
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

/**
 * @brief The core class that encapsulates the logic for actually running the training.
 */
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

    std::unique_ptr<TensorBoardLogger> logger_;

    /**
     * @brief Sets up the dimensions of tensors and buffers used in the training loop.
     * 
     * This method will handle initializing every factor used in training corresponding 
     * to any values set in its config.
     */
    void setup_dimensions_and_buffers();

    /**
     * @brief Attempts to find a load a checkpoint for a given bot.
     * 
     * @param dir - The directory to search for checkpoints with the format: <bot name>_<total steps trained>.pt
     * @return int64_t - If a checkpoint was found then the total steps trained, otherwise -1.
     */
    int64_t load_latest_checkpoint(const std::string& dir);
    
    /**
     * @brief Runs bot training and handles all associated processes.
     * 
     * @param num_updates - The number of updates to run the training for.
     * @param stop_flag - The atomic flag to indicate that the training loop should stop and save.
     * @param checkpoint_dir - The directory to save a checkpoint to.
     */
    void run_training(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir);

    /**
     * @brief Handles running a single environment as a render of the bot in RocketSim.
     * 
     * @param stop_flag - The atomic flad to indicate that the render loop should stop.
     */
    void run_render(const std::atomic<bool>& stop_flag);

public:
    explicit PPORunner(const RunnerConfig& config);

    /**
     * @brief Saves a checkpoint of the bot.
     * 
     * @param dir - The directory to save the checkpoint to.
     */
    void save_checkpoint(const std::string& dir) const;

    /**
     * @brief Chooses between running render or training based on the config.
     * 
     * @param num_updates - The number of updates to run the training for.
     * @param stop_flag - The atomic flag to indicate that the training loop should stop and save.
     * @param checkpoint_dir - The directory to save a checkpoint to.
     */
    void run(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir);
};

}; // namespace buta_ppo::rl