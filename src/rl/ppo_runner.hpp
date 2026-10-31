// buta-ppo/src/rl/ppo_runner.hpp
#pragma once

#include "buta_ppo/ffi.h"
#include "ppo_trainer.hpp"
#include "actor_critic.hpp"
#include "pfsp.hpp"
#include "state/state_setter.hpp"
#include "reward/reward_manager.hpp"
#include "metric/metric_manager.hpp"
#include "env/episode_limits.hpp"
#include "external/external_bot.hpp"
#include "gauntlet.hpp"

#include <torch/torch.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>

#include <cstddef>
#include <stdexcept>
#include <functional>
#include <memory>
#include <atomic>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// Forward declarations.
class TensorBoardLogger;
namespace buta_ppo::env { class VecEnv; struct BatchedStepResult; }
namespace buta_ppo::rl  { class RolloutBuffer; class OpponentPool; }

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
 * @brief The values one TensorBoard write logged, by tag (see docs/telemetry.md), for `RunnerConfig::stop_condition`.
 */
struct TelemetryPoint {
    int64_t steps = 0;        // The lifetime steps it was logged at.
    // The steps a log window covered: its counts (`Episode/Goals`, the truncations, `PFSP/Games`) are totals over them.
    // 0 for an evaluation's values (`Skill/`).
    int64_t window_steps = 0;
    std::unordered_map<std::string, double> values;

    /**
     * @brief The value logged under `tag`, if this write logged one.
     */
    [[nodiscard]] std::optional<double> find(const std::string& tag) const {
        const auto it = values.find(tag);
        return it == values.end() ? std::nullopt : std::optional<double>(it->second);
    }
};

struct RunnerConfig;

/**
 * @brief Decides from telemetry whether training should stop: a reason to stop, or nothing to carry on.
 *
 * Gets the values just logged and the config training runs with (its reward entries, `ppo_cfg.gae_gamma`).
 */
using StopCondition = std::function<std::optional<std::string>(const TelemetryPoint&, const RunnerConfig&)>;

/**
 * @brief The core config for a bot trained with this framework.
 * 
 * Contains all of the configuration values for a bot with defaults in place 
 * allowing the user to have a baseline before customizing it to their needs.
 * 
 * NOTE: `setter_distributions` MUST be set by the user.
 * NOTE: `num_minibatches` is HIGHLY hardware and network layer setup dependent. Experiment 
 * on your own. Start small on steps per iteration and high on num minibatches. For me with 
 * 12gb VRAM I was able to comfortably get away with 125'000 steps per iter and 1
 * minibatch. I honestly dont know what happens if you exceed available VRAM but it should
 * crash immediately and just be the program... probably.  
 */
struct RunnerConfig {
    std::string bot_name = "default";
    bool render = false;

    // Render only: external bots (see `src/external/bots.hpp`) that drive a whole team instead of the policy, e.g.
    // `{{create_nexto_bot("src/external/nexto/nexto-model.pt"), ffi::Team::Orange}}`. Teams not listed play the policy.
    std::vector<external::ExternalSide> render_external_bots = {};

    // Render only: the policy plays its most likely action, as the deployed bot does, instead of sampling like in
    // training.
    bool render_best_action = true;

    // Instead of training, measures thread counts, CPU pinning and env counts on this machine and prints the fastest
    // as config values to paste in. It runs this config's match layouts and network with fresh weights, under a light
    // workload (a fresh policy, mostly airborne) and a heavy one (scripted cars chasing the ball, like a bot that keeps
    // touching it), since a bot's speed swings between the two as it learns. Nothing is loaded, saved, or logged.
    bool benchmark = false;

    // Instead of training, calibrates the Nexto ladder precisely (`eval_cfg.nexto_calibration_bank_size` and
    // `nexto_calibration_rungs_above`; about 20 minutes with the defaults). With the default kickoff jitter and game cap
    // it regenerates the shipped ladder, only needed after changing Nexto's model, the ladder's rungs or the eval
    // kickoff setter. With other values, which get a quick, rougher ladder at startup by themselves, it writes a precise
    // one to `<checkpoint dir>/nexto_ladder.txt`, where evaluation looks for it. A changed `eval_cfg.nexto_ladder_path`
    // is written instead. A new ladder restarts the Elo curve, so calibrate before training with the new settings.
    bool calibrate_nexto_ladder = false;

    std::vector<MatchDistribution> match_distributions = {
        {1, 1, 1.0f}
    };

    std::vector<state::StateSetterDistribution> setter_distributions;

    // Besides a goal, an episode ends when no car has touched the ball for this long, or when it has lasted this
    // long. Both are truncations, so the critic's value of the last state stands in for the rest of the episode.
    // 0 turns a limit off.
    float no_touch_timeout_seconds = 10.0f;
    float episode_time_limit_seconds = 150.0f;
    std::vector<reward::RewardEntry> reward_entries;
    std::vector<metric::MetricEntry> metric_entries = {};

    // Total envs. 0 picks the env count whose steps per agent come closest to target_steps_per_agent.
    // Workers take envs one at a time, so it doesn't need to divide by the thread count. Training needs at least 2.
    size_t num_envs = 0;
    // Env worker threads. 0 uses one per CPU in worker_cpus, or when it's empty, one per CPU this process may run on
    // except one left for the main thread.
    size_t num_threads = 0;

    // Optional CPU pinning, by logical CPU id (see `lscpu -e`). Empty leaves those threads unpinned.
    // Pinning helps only when the main thread limits the rollout, shown by a low `Performance/Rollout_Physics_Wait`
    // (about 0.1), e.g. on a hybrid CPU: the main thread gets a P-core to itself and the workers get every other CPU.
    // When physics is the limit (about 0.25 and up), taking CPUs away from the workers slows training down.
    std::vector<int> main_thread_cpus = {}; // The thread running inference and the step bookkeeping.
    std::vector<int> worker_cpus = {};      // The env worker threads, which share these CPUs.

    size_t max_players_per_team = 3;
    int ticks_per_step = 8;

    size_t target_steps_per_update = 50'000; // Good starting values to guage your hardware with your network size.
    size_t num_minibatches = 4;
    // The steps each learner agent collects per update (target_steps_per_update / learner agents). Sets the env count
    // when num_envs is 0. Fewer steps per agent means more envs, which is usually faster, but each update then learns
    // from shorter stretches of play. The benchmark aims its env counts at the resulting env count.
    size_t target_steps_per_agent = 100;

    // Lifetime steps between checkpoints saved during training. 0 only saves when training ends or is interrupted.
    int64_t steps_per_checkpoint = 100'000'000;

    // Lifetime steps between TensorBoard logs. Each log aggregates every update since the previous one.
    // 0 logs every update, as does any value below the actual steps per update.
    int64_t steps_per_log = 0;

    // TensorBoard's name for this run. TensorBoard names a run by its event file's directory under the log dir, so
    // the event file goes to `logs/<run_name>/<bot name>.tfevents`, and `tensorboard --logdir logs` shows every run
    // side by side. May have subdirectories ("porkchop/run2"). Empty writes `logs/<bot name>.tfevents`, a run named
    // ".". Changing it starts a new event file; checkpoints don't depend on it.
    std::string run_name;

    // Lifetime steps to train up to. Training stops after the update that reaches them and saves a checkpoint, and a
    // start at or past them trains nothing. 0 trains until the update count given to `run` is used up or training is
    // stopped. Training plans (`training_plan.hpp`) set it to each stage's end.
    int64_t train_until_steps = 0;

    // Checks telemetry during training, to stop a run that went wrong while nobody was watching. Called after each
    // TensorBoard log (`steps_per_log`) and each evaluation. Returning a reason stops training as Ctrl+C does: the
    // update finishes and a checkpoint is saved, and a training plan ends there too. The reason is printed and appended
    // to `<bot name>.stops.txt` next to the event file (see `run_name`). Empty never stops.
    StopCondition stop_condition;

    ActorCriticConfig ac_cfg;
    PPOConfig ppo_cfg;
    PFSPConfig pfsp_cfg;
    EvalConfig eval_cfg;

    /**
     * @brief The total envs: `num_envs`, or the env count closest to `target_steps_per_agent` when it is 0.
     *
     * @throws std::invalid_argument If both are 0.
     */
    [[nodiscard]] size_t total_envs() const;
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
 * @brief The learner agents a total env count gets, as VecEnv lays them out. PFSP opponents aren't counted.
 *
 * @param envs The total envs.
 * @param config The config with the match distributions and PFSP settings.
 */
[[nodiscard]] size_t learner_agents_for(size_t envs, const RunnerConfig& config);

/**
 * @brief The lifetime steps of a bot's latest checkpoint, the one training resumes from.
 *
 * @param dir The checkpoint directory.
 * @param bot_name The bot whose `<bot name>_<steps>/` checkpoints to look for.
 * @return The steps, or 0 without a checkpoint (training starts fresh).
 */
[[nodiscard]] int64_t latest_checkpoint_steps(const std::string& dir, const std::string& bot_name);

/**
 * @brief The core class that encapsulates the logic for actually running the entire training process.
 */
class PPORunner {
private:
    RunnerConfig config_;
    torch::Device device_;
    env::EpisodeLimits episode_limits_; // The config's episode limits in physics ticks.

    std::unique_ptr<env::VecEnv> vec_env_;
    ActorCritic actor_critic_{nullptr};
    std::unique_ptr<PPOTrainer> trainer_;
    std::unique_ptr<RolloutBuffer> buffer_;

    size_t total_agents_{0};   // Every agent's row of the batched env buffers and actions.
    size_t learner_agents_{0}; // Agents acting with the learning policy, rows [0, learner agents). Only they are trained on.
    size_t buffer_size_{0};
    int64_t total_steps_per_update_{0};
    size_t obs_size_{0};

    torch::Tensor actions_cpu_;
    torch::Tensor step_obs_gpu_;
    torch::Tensor step_masks_gpu_;
    torch::Tensor step_rewards_gpu_;
    torch::Tensor step_term_gpu_;
    torch::Tensor step_trunc_gpu_;
    torch::Tensor step_dones_gpu_;

    // Staging buffers for deferred truncation critic forward pass.
    std::vector<int64_t> deferred_idx_;
    std::vector<float> deferred_obs_;

    // PFSP opponent slots: fixed modules whose weights are overwritten in place every update, so the inference
    // graphs that read them stay valid. Opponent obs and masks are staged on the GPU in their own buffers,
    // indexed by opponent row minus the learner agent count.
    std::vector<ActorCritic> opponent_slots_;
    // Each slot's inference runs on its own stream, so it overlaps with the learner's and the other slots'.
    std::vector<at::cuda::CUDAStream> opponent_streams_;

    // Past learner versions the opponent slots are loaded from. Null when PFSP is disabled.
    std::unique_ptr<OpponentPool> opponent_pool_;
    int64_t next_snapshot_step_{0};
    torch::Tensor opp_obs_gpu_;
    torch::Tensor opp_masks_gpu_;

    // Evaluation gauntlets. Null when evaluation is off, and in render and benchmark runs.
    std::unique_ptr<Gauntlet> gauntlet_;
    int64_t next_eval_step_{0};

    /**
     * @brief Runs a gauntlet on the current learner, then logs and prints its results.
     */
    void run_evaluation();

    /**
     * @brief One env group's slice of the rollout: its agent rows and its inference CUDA graph.
     *
     * The tensors are row views of the runner's full-batch buffers, so every group writes disjoint rows.
     * Graph outputs live in the graph's memory pool and are rewritten by every replay.
     */
    struct RolloutGroup {
        int64_t learner_begin = 0;
        int64_t learner_count = 0;

        // Row views.
        torch::Tensor obs_src;       // VecEnv's pinned obs rows that the graph uploads from.
        torch::Tensor masks_src;     // VecEnv's pinned action mask rows that the graph uploads from.
        torch::Tensor actions_cpu;
        torch::Tensor obs_gpu;
        torch::Tensor masks_gpu;
        torch::Tensor rewards_gpu;
        torch::Tensor term_gpu;
        torch::Tensor trunc_gpu;
        torch::Tensor dones_gpu;

        // The group's opponent rows for one opponent slot, as views like the learner rows above.
        struct OpponentRows {
            torch::Tensor obs_src;
            torch::Tensor masks_src;
            torch::Tensor obs_gpu;
            torch::Tensor masks_gpu;
            torch::Tensor actions_cpu;
        };
        std::vector<OpponentRows> opponents; // One per opponent slot.

        // Inference graph and its outputs.
        at::cuda::CUDAGraph graph;
        bool captured = false;
        torch::Tensor actions_gpu;
        torch::Tensor log_probs_gpu;
        torch::Tensor values_gpu;

        // Insert graph: uploads a step's results and writes the step into the buffer. The step it writes lives on the
        // device, and the graph advances it, so one capture serves every step. `buffer_step` is the host's count of
        // the same steps, checked against the device's at the end of each rollout.
        at::cuda::CUDAGraph insert_graph;
        bool insert_captured = false;
        torch::Tensor buffer_step_gpu;
        size_t buffer_step = 0;
        torch::Tensor rewards_src;   // VecEnv's pinned result rows that the insert graph uploads from.
        torch::Tensor term_src;
        torch::Tensor trunc_src;
    };
    std::vector<std::unique_ptr<RolloutGroup>> rollout_groups_;

    int64_t global_step_{0};

    // Periodic checkpoints. The last saved step lets the final save skip a checkpoint that was just written.
    int64_t next_checkpoint_step_{0};
    int64_t last_checkpoint_step_{-1};

    std::unique_ptr<TensorBoardLogger> logger_;

    // Benchmark runners measure one candidate config each, quietly and without logging.
    bool benchmark_child_ = false;
    // VecEnv's full batched obs buffer, which always holds every agent's latest obs. Set by `prepare_rollouts`.
    torch::Tensor batched_obs_;

    // Everything since the last TensorBoard log, written as one aggregated point per log. Rewards and metrics
    // keep accumulating inside the envs instead, until `update_telemetry_breakdowns` reads them at log time.
    struct LogWindow {
        int64_t steps = 0;
        int64_t updates = 0;
        double update_seconds = 0.0;
        double rollout_seconds = 0.0;
        double physics_wait_seconds = 0.0; // Part of the rollout the main thread spent waiting for the env workers.
        uint64_t goals = 0;
        uint64_t no_touch_truncations = 0;
        uint64_t time_limit_truncations = 0;
        uint64_t pfsp_wins = 0;
        uint64_t pfsp_games = 0;
        std::unordered_map<std::string, double> train_metrics; // Summed over updates.
    };
    LogWindow log_window_;
    int64_t next_log_step_{0};
    // The values of the TensorBoard write in progress (a log window or an evaluation), for the stop condition.
    TelemetryPoint telemetry_;
    std::optional<std::string> stop_reason_;
    /**
     * @brief Logs a TensorBoard scalar and records it in `telemetry_`.
     */
    template <class T>
    void log_scalar(const std::string& tag, int64_t step, T value);
    /**
     * @brief Runs the stop condition on `telemetry_`, and on a reason to stop, keeps it, prints it and appends it to
     * `<bot name>.stops.txt` in `log_dir()`. Training stops before its next update.
     */
    void check_stop_condition();

    // Rollout bottleneck warnings, see `check_rollout_bottleneck`.
    size_t allowed_cpus_{0};                  // CPUs this process may run on, read before any thread is pinned.
    std::vector<double> recent_physics_wait_; // The last updates' Phys Wait fractions, oldest overwritten first.
    size_t next_physics_wait_{0};
    bool warned_physics_bound_{false};
    bool warned_main_bound_{false};

    // Every warning of this run, so fast scrolling updates can't hide them, see `warn`.
    struct RunWarning {
        std::string message;
        std::string reminder; // Short label repeated under every update.
        int update;           // The update it was raised in, 0 before training.
    };
    std::vector<RunWarning> warnings_;
    int current_update_{0};

    /**
     * @brief Prints a warning and keeps it for the reminder line under every update and the end of run summary.
     *
     * @param message The full warning, printed now and again at the end of the run.
     * @param reminder A short label for the reminder line. Warnings sharing a label show it once.
     */
    void warn(const std::string& message, const std::string& reminder);

    /**
     * @brief Prints the reminder line of the warnings so far under an update. Prints nothing without warnings.
     */
    void print_warning_reminders() const;

    /**
     * @brief Prints every warning of the run again, at the end of training.
     */
    void print_warning_summary() const;

    /**
     * @brief Checks the reward entries against the match layouts. Throws on invalid entries and prints warnings
     * for settings that likely don't do what was intended.
     */
    void validate_rewards();

    /**
     * @brief Checks the CPU pinning config. Throws on CPUs this process can't run on and warns when the main
     * thread's CPUs are shared with the workers.
     */
    void validate_cpu_pinning();

    /**
     * @brief Warns, once per run, when the rollout's bottleneck has gone on for a while with CPUs that could help.
     *
     * Looks at the median Phys Wait of the last updates, so it ignores the swings of a bot's play. Physics as the
     * limit while CPUs have no worker, or the main thread as the limit while it shares every CPU with the workers,
     * both mean the CPU setup leaves speed on the table. Configs the benchmark recommends trigger neither.
     *
     * @param physics_wait The share of this update's rollout the main thread spent waiting for the env workers.
     */
    void check_rollout_bottleneck(double physics_wait);

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
     * @brief Overwrites an opponent slot's weights in place, keeping the addresses its inference graphs read.
     *
     * Must not run while a rollout inference graph replay is in flight.
     *
     * @param slot The opponent slot to load.
     * @param params The weights in `parameters()` order of an ActorCritic with the same config. Undefined tensors
     *               are skipped, leaving the slot's weight as is.
     */
    void load_opponent_slot(size_t slot, const std::vector<torch::Tensor>& params);

    /**
     * @brief The rollout inference work captured in a group's CUDA graph.
     *
     * Uploads the group's obs and masks, samples actions, and queues the actions' copy into its `actions_cpu` rows.
     * Does the same for each opponent slot's rows of the group, sampling from that slot's policy on the slot's stream.
     * Inside a capture, each slot becomes a parallel branch of the graph.
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
     * @brief The insert work captured in a group's insert graph.
     *
     * Uploads the group's rewards and episode ends, writes the step's obs, masks, inference outputs and results into
     * the buffer at the group's device step, and advances that step.
     */
    void insert_graph_logic(RolloutGroup& group);

    /**
     * @brief Stores a group's truncated agents for bootstrapping, and inserts the step into the buffer by replaying
     * the group's insert graph, capturing it on first use. Doesn't wait for the GPU.
     *
     * Must run before the group's next `infer_group`, which overwrites the obs rows and graph outputs it inserts.
     *
     * @param group The group whose step finished.
     * @param step The buffer step to write. Must be the group's next step, which the insert graph writes.
     * @param step_res The group's rows of the step result. Must be the same rows on every call.
     */
    void finish_group_step(RolloutGroup& group, size_t step, const env::BatchedStepResult& step_res);

    /**
     * @brief Resets the envs and sets up the rollout groups on VecEnv's batched buffers. Run once before rollouts.
     */
    void prepare_rollouts();

    /**
     * @brief Collects one update's rollout into the buffer, including the deferred truncation bootstraps.
     *
     * Needs `prepare_rollouts` and the opponent slots loaded first.
     *
     * @return The seconds the main thread spent waiting for the env workers.
     */
    double collect_rollout();

    /**
     * @brief Bootstraps the rollout's cutoff from the critic and computes the buffer's returns and advantages.
     */
    void compute_advantages();

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
     * @brief Where the event file and the stops file go: `logs/`, or `logs/<run_name>/`.
     */
    [[nodiscard]] std::string log_dir() const;
    /**
     * @brief The config saved with every model: the network config and the obs settings from the runner config.
     */
    [[nodiscard]] ModelConfig model_config() const;
    
    /**
     * @brief Runs bot training and handles all associated processes.
     * 
     * @param num_updates The number of updates to run the training for.
     * @param stop_flag The atomic flag to indicate that the training loop should stop and save.
     * @param checkpoint_dir The directory to save a checkpoint to.
     */
    void run_training(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir);

    /**
     * @brief A benchmark runner for one candidate config. Prints nothing while building and never logs.
     */
    PPORunner(const RunnerConfig& config, bool benchmark_child);

    /**
     * @brief Measures thread counts, CPU pinning and env counts, then prints the results and the recommended config.
     *
     * @param stop_flag Stops the benchmark early, printing what was measured so far.
     */
    void run_benchmark(const std::atomic<bool>& stop_flag);

    /**
     * @brief Calibrates the Nexto ladder (see `RunnerConfig::calibrate_nexto_ladder`): the shipped file for the default
     * settings, else `<checkpoint_dir>/nexto_ladder.txt`, unless `eval_cfg.nexto_ladder_path` was changed.
     */
    void run_ladder_calibration(const std::string& checkpoint_dir);

    /**
     * @brief Timing of one benchmark sample.
     */
    struct UpdateSample {
        std::vector<double> update_seconds;  // One per measured update: rollout, advantages and training.
        std::vector<double> rollout_seconds;
        std::vector<double> physics_wait;    // Fraction of each update's rollout spent waiting for the env workers.
    };

    /**
     * @brief Times training updates of this benchmark runner under one workload, the same way `run_training` runs them.
     *
     * The rollouts and the training between them both count: back-to-back rollouts alone run at a different speed
     * than in training on some CPU setups.
     *
     * @param heavy Whether to run the heavy workload (see `RunnerConfig::benchmark`).
     * @param warmup Updates run first and not measured.
     * @param measured Updates measured.
     */
    UpdateSample measure_updates(bool heavy, int warmup, int measured);

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
     * Writes `model.pt` (with its `ModelConfig`) and `optimizer.pt` into <dir>/<bot name>_<total steps trained>/.
     *
     * @param dir The directory to save the checkpoint to.
     */
    void save_checkpoint(const std::string& dir) const;

    /**
     * @brief Writes the log window to TensorBoard as averages over its steps and updates, then clears it.
     *
     * Reads the envs' accumulated reward and metric telemetry, so it must not run while an env group is stepping.
     */
    void write_log_window();

    /**
     * @brief Chooses between running render or training based on the config.
     * 
     * @param num_updates The number of updates to run the training for.
     * @param stop_flag The atomic flag to indicate that the training loop should stop and save.
     * @param checkpoint_dir The directory to save a checkpoint to.
     */
    void run(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir);
    /**
     * @brief Why the stop condition stopped training, if it did.
     */
    [[nodiscard]] const std::optional<std::string>& stop_reason() const { return stop_reason_; }
};

} // namespace buta_ppo::rl