// buta-ppo/src/rl/ppo_runner.hpp
#pragma once

#include "buta_ppo/ffi.h"
#include "ppo_trainer.hpp"
#include "actor_critic.hpp"
#include "state/state_setter.hpp"
#include "reward/reward_manager.hpp"

#include <torch/torch.h>
#include <ATen/cuda/CUDAGraph.h>

#include <cstddef>
#include <stdexcept>
#include <memory>
#include <atomic>
#include <string>
#include <tuple>
#include <vector>

// Forward declarations.
class TensorBoardLogger;
namespace buta_ppo::env { class VecEnv; struct BatchedStepResult; }
namespace buta_ppo::rl  { class RolloutBuffer; }

namespace buta_ppo::rl {

/**
 * @brief Formatter for match distributions.
 * 
 * Allows the user to neatly define match distibutions along 
 * with their associated weight to be represented within the envs.
 */
struct MatchDistribution {
    size_t blue_players = 1;
    size_t orange_players = 1;
    float weight = 1.0f;

    /**
     * @brief Total players in this match disctribution.
     * 
     * @return Total players across both teams.
     */
    [[nodiscard]] size_t total_players() const {
        return blue_players + orange_players;
    }

    /**
     * @brief Translates from this struct format to a team layout using the FFI Team enum.
     * 
     * @return The vector of all team member's Team enum.
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
 * 
 * NOTE: `setter_distributions` MUST be set by the user.
 * NOTE: `num_minibatches` is HIGHLY hardware and network layer setup dependent. Experiment 
 * on your own. Start small on steps per iteration and high on num minibatches. For me with 
 * 12gb VRAM I was able to comfortably get away with 50'000 steps per iter and 1
 * minibatch. I honestly dont know what happens if you exceed available VRAM but it should
 * crash immediately and just be the program... probably.  
 */
struct RunnerConfig {
    std::string bot_name = "default";
    bool render = false;

    std::vector<MatchDistribution> match_distributions = {
        {1, 1, 1.0f}
    };

    std::vector<state::StateSetterDistribution> setter_distributions;
    std::vector<reward::RewardEntry> reward_entries;

    size_t num_envs_per_thread = 20;
    size_t num_threads = 12;
    size_t max_players_per_team = 3;
    int ticks_per_step = 8;

    size_t target_steps_per_update = 50'000; // Good starting values to guage your hardware with your network size.
    size_t num_minibatches = 4;

    ActorCriticConfig ac_cfg;
    PPOConfig ppo_cfg;
};

/**
 * @brief Given the desired match distributions, evenly allocate environments to represent them.
 * 
 * @param target_total_envs The total envs to have.
 * @param distributions The distributions and weights to allocate envs to.
 * @return A vector containing the number of envs correlating to the match distributions.
 */
[[nodiscard]] inline std::vector<size_t> compute_env_counts(
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
        const float norm_weight = distributions[i].weight / weight_sum;
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
 * @brief The core class that encapsulates the logic for actually running the entire training process.
 */
class PPORunner {
private:
    RunnerConfig config_;
    torch::Device device_;

    std::unique_ptr<env::VecEnv> vec_env_;
    ActorCritic actor_critic_{nullptr};
    std::unique_ptr<PPOTrainer> trainer_;
    std::unique_ptr<RolloutBuffer> buffer_;

    size_t total_agents_{0};
    size_t buffer_size_{0};
    int64_t total_steps_per_update_{0};

    torch::Tensor actions_cpu_;
    torch::Tensor step_obs_gpu_;
    torch::Tensor step_masks_gpu_;
    torch::Tensor step_rewards_gpu_;
    torch::Tensor step_term_gpu_;
    torch::Tensor step_trunc_gpu_;
    torch::Tensor step_terminal_obs_gpu_;
    torch::Tensor step_dones_gpu_;

    /**
     * @brief One env group's slice of the rollout: its agent rows and its inference CUDA graph.
     *
     * The tensors are row views of the runner's full-batch buffers, so every group writes disjoint rows.
     * Graph outputs live in the graph's memory pool and are rewritten by every replay.
     */
    struct RolloutGroup {
        int64_t agent_begin = 0;
        int64_t agent_count = 0;

        // Row views.
        torch::Tensor obs_src;       // VecEnv's pinned obs rows that the graph uploads from.
        torch::Tensor masks_src;     // VecEnv's pinned action mask rows that the graph uploads from.
        torch::Tensor actions_cpu;
        torch::Tensor obs_gpu;
        torch::Tensor masks_gpu;
        torch::Tensor rewards_gpu;
        torch::Tensor term_gpu;
        torch::Tensor trunc_gpu;
        torch::Tensor terminal_obs_gpu;
        torch::Tensor dones_gpu;

        // Inference graph and its outputs.
        at::cuda::CUDAGraph graph;
        bool captured = false;
        torch::Tensor actions_gpu;
        torch::Tensor log_probs_gpu;
        torch::Tensor values_gpu;
    };
    std::vector<std::unique_ptr<RolloutGroup>> rollout_groups_;

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
     * @brief Creates the per-group row views for every VecEnv env group.
     *
     * @param obs VecEnv's full batched obs buffer.
     * @param masks VecEnv's full batched action mask buffer.
     */
    void setup_rollout_groups(const torch::Tensor& obs, const torch::Tensor& masks);

    /**
     * @brief The rollout inference work captured in a group's CUDA graph.
     *
     * Uploads the group's obs and masks, samples actions, and queues the actions' copy into its `actions_cpu` rows.
     *
     * @return The sampled actions, their log probs, and the state values.
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> inference_graph_logic(RolloutGroup& group);

    /**
     * @brief Samples actions for a group's current obs by replaying its inference graph, capturing it on first use.
     *
     * Blocks until the actions are in the group's `actions_cpu` rows. The group's graph outputs hold the results
     * until its next call.
     *
     * @param group The group to infer.
     * @param obs The group's rows of VecEnv's obs buffer. Must be the same rows on every call.
     * @param masks The group's rows of VecEnv's action mask buffer. Must be the same rows on every call.
     */
    void infer_group(RolloutGroup& group, const torch::Tensor& obs, const torch::Tensor& masks);

    /**
     * @brief Uploads a group's step results, bootstraps its truncated agents, and inserts the step into the buffer.
     *
     * Must run before the group's next `infer_group`, which overwrites the obs rows and graph outputs it inserts.
     *
     * @param group The group whose step finished.
     * @param step The buffer step to write.
     * @param step_res The group's rows of the step result.
     */
    void finish_group_step(RolloutGroup& group, size_t step, const env::BatchedStepResult& step_res);

    /**
     * @brief Attempts to find and load the latest checkpoint (model and optimizer) for a given bot.
     *
     * Checkpoints are directories with the format: <bot name>_<total steps trained>/ containing
     * `model.pt` and `optimizer.pt`. A missing or mismatched optimizer state falls back to a fresh optimizer.
     *
     * @param dir The directory to search for checkpoints.
     * @return If a checkpoint was found then the total steps trained, otherwise 0.
     */
    int64_t load_latest_checkpoint(const std::string& dir);
    
    /**
     * @brief Runs bot training and handles all associated processes.
     * 
     * @param num_updates The number of updates to run the training for.
     * @param stop_flag The atomic flag to indicate that the training loop should stop and save.
     * @param checkpoint_dir The directory to save a checkpoint to.
     */
    void run_training(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir);

    /**
     * @brief Handles running a single environment as a render of the bot in RocketSim.
     * 
     * @param stop_flag The atomic flag to indicate that the render loop should stop.
     */
    void run_render(const std::atomic<bool>& stop_flag);

public:
    explicit PPORunner(const RunnerConfig& config);
    ~PPORunner();

    /**
     * @brief Saves a checkpoint of the bot's model and optimizer state.
     *
     * Writes `model.pt` and `optimizer.pt` into <dir>/<bot name>_<total steps trained>/.
     *
     * @param dir The directory to save the checkpoint to.
     */
    void save_checkpoint(const std::string& dir) const;

    /**
     * @brief Chooses between running render or training based on the config.
     * 
     * @param num_updates The number of updates to run the training for.
     * @param stop_flag The atomic flag to indicate that the training loop should stop and save.
     * @param checkpoint_dir The directory to save a checkpoint to.
     */
    void run(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir);
};

} // namespace buta_ppo::rl