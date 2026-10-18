// buta-ppo/src/rl/ppo_runner.cpp
#include "ppo_runner.hpp"
#include "rollout_buffer.hpp"
#include "opponent_pool.hpp"
#include "tensorboard_logger.h"
#include "env/vec_env.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <memory>
#include <stdexcept>

namespace fs = std::filesystem;

namespace buta_ppo::rl {

namespace {
// File names inside each `<bot name>_<steps>/` checkpoint directory.
constexpr const char* MODEL_FILE = "model.pt";
constexpr const char* OPTIMIZER_FILE = "optimizer.pt";
// Suffix of the PFSP pool's directory, `<bot name>_pool/`, next to the checkpoints.
// Checkpoint loading skips it, since "pool" isn't a step count.
constexpr const char* POOL_DIR_SUFFIX = "_pool";
} // namespace

PPORunner::PPORunner(const RunnerConfig& config)
    : config_(config), device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
    
    if (torch::cuda::is_available()) {
        std::cout << "CUDA detected. Running on GPU.\n";
        at::globalContext().setUserEnabledCuDNN(true);
        at::globalContext().setBenchmarkCuDNN(true);

        // TF32 matmuls for the fp32 rollout inference (training matmuls run in bf16 under autocast).
        // TF32 keeps fp32 range and accumulation with a 10-bit mantissa, still above bf16's 7.
        at::globalContext().setAllowTF32CuBLAS(true);
    } else {
        throw std::runtime_error("CUDA not found. CPU is currently not supported.");
    }

    validate_rewards();

    fs::create_directories("logs");
    const std::string log_file = "logs/" + config_.bot_name + ".tfevents";
    TensorBoardLoggerOptions logger_options_{};
    logger_options_.resume_ = true;
    logger_ = std::make_unique<TensorBoardLogger>(log_file.c_str(), logger_options_);

    setup_dimensions_and_buffers();
}

PPORunner::~PPORunner() = default;

int64_t PPORunner::load_latest_checkpoint(const std::string& dir) {
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
        return 0;
    }

    fs::path latest_dir;
    int64_t max_steps = -1;

    // Scan for checkpoint directories named <bot name>_<steps> that contain a model.
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_directory() || !fs::exists(entry.path() / MODEL_FILE)) continue;

        const std::string name = entry.path().filename().string();
        const size_t delim_pos = name.find_last_of('_');
        if (delim_pos == std::string::npos || name.substr(0, delim_pos) != config_.bot_name) continue;

        // The whole suffix must be the step count, so in-progress `<bot name>_<steps>.tmp` saves are skipped.
        int64_t steps = 0;
        const char* first = name.data() + delim_pos + 1;
        const char* last = name.data() + name.size();
        const auto [ptr, ec] = std::from_chars(first, last, steps);
        if (ec != std::errc{} || ptr != last || first == last) continue;

        if (steps > max_steps) {
            max_steps = steps;
            latest_dir = entry.path();
        }
    }

    if (max_steps < 0) {
        std::cout << "No valid checkpoints found for bot '" << config_.bot_name << "'. Starting fresh training.\n";
        return 0;
    }

    torch::load(actor_critic_, (latest_dir / MODEL_FILE).string());
    std::cout << "Latest Model Found: " << latest_dir.string() << " (Lifetime Steps: " << max_steps << ")\n";

    const fs::path optimizer_path = latest_dir / OPTIMIZER_FILE;
    if (fs::exists(optimizer_path) && trainer_->load_optimizer(optimizer_path.string())) {
        std::cout << "Optimizer state restored.\n";
    } else {
        std::cout << "No usable optimizer state found. Starting with a fresh optimizer.\n";
    }
    return max_steps;
}

void PPORunner::validate_rewards() const {
    // Only layouts that get envs matter.
    size_t largest_team = 0;
    bool has_empty_team = false;
    for (const auto& dist : config_.match_distributions) {
        if (dist.weight <= 0.0f) continue;
        largest_team = std::max({largest_team, dist.blue_players, dist.orange_players});
        has_empty_team |= dist.blue_players == 0 || dist.orange_players == 0;
    }

    for (const std::string& warning : reward::validate_reward_entries(config_.reward_entries, largest_team, has_empty_team)) {
        std::cerr << "[WARN] " << warning << "\n";
    }
}

void PPORunner::setup_dimensions_and_buffers() {

    if (config_.render) {
        config_.num_envs_per_thread = 1;
        config_.num_threads = 1;
        config_.num_minibatches = 1;
    }

    vec_env_ = std::make_unique<env::VecEnv>(
        config_.num_envs_per_thread, 
        config_.match_distributions,
        config_.setter_distributions,
        config_.reward_entries,
        config_.metric_entries,
        config_.num_threads, 
        config_.ticks_per_step, 
        config_.max_players_per_team,
        config_.render,
        config_.pfsp_cfg
    );

    total_agents_ = vec_env_->get_total_agents();
    learner_agents_ = vec_env_->get_learner_agents();
    obs_size_ = vec_env_->get_single_obs_size();
    const size_t action_space_size = vec_env_->get_action_space_size();

    if (config_.render) {
        config_.target_steps_per_update = learner_agents_;
    }

    buffer_size_ = (config_.target_steps_per_update + learner_agents_ - 1) / learner_agents_;
    while ((buffer_size_ * learner_agents_) % config_.num_minibatches != 0) {
        buffer_size_++;
    }

    // Initialize staging buffers for deferred truncation critic forward pass.
    // Reserve about 2× the rows a typical update stages; vectors keep capacity after clear(), so a larger update grows them once.
    deferred_idx_.reserve(2 * learner_agents_);
    deferred_obs_.reserve(2 * learner_agents_ * obs_size_);

    total_steps_per_update_ = static_cast<int64_t>(buffer_size_ * learner_agents_);
    config_.ppo_cfg.mini_batch_size = total_steps_per_update_ / config_.num_minibatches;

    config_.ac_cfg.obs_size = static_cast<int64_t>(obs_size_);
    config_.ac_cfg.action_size = static_cast<int64_t>(action_space_size);
    
    actor_critic_ = ActorCritic(config_.ac_cfg);
    actor_critic_->to(device_);

    trainer_ = std::make_unique<PPOTrainer>(config_.ppo_cfg, actor_critic_, device_);

    // Opponent slots only run inference, so their weights never need gradients.
    for (size_t slot = 0; slot < vec_env_->get_num_opponent_slots(); slot++) {
        ActorCritic opponent(config_.ac_cfg);
        opponent->to(device_);
        for (auto& param : opponent->parameters()) {
            param.requires_grad_(false);
        }
        opponent_slots_.push_back(opponent);
        opponent_streams_.push_back(at::cuda::getStreamFromPool());
    }
    if (!opponent_slots_.empty()) {
        opponent_pool_ = std::make_unique<OpponentPool>(config_.pfsp_cfg, config_.ac_cfg);
    }
    
    buffer_ = std::make_unique<RolloutBuffer>(
        buffer_size_, learner_agents_, obs_size_, action_space_size, device_
    );

    // Actions cover every row, since VecEnv reads each agent's action from its row.
    // The step tensors only hold learner rows.
    const int64_t t_agents = static_cast<int64_t>(total_agents_);
    const int64_t l_agents = static_cast<int64_t>(learner_agents_);
    const auto cpu_int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    actions_cpu_ = torch::empty({t_agents}, cpu_int_opts);

    const auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    step_obs_gpu_ = torch::empty({l_agents, static_cast<int64_t>(obs_size_)}, float_opts);
    step_masks_gpu_ = torch::empty({l_agents, static_cast<int64_t>(action_space_size)}, float_opts);
    step_rewards_gpu_ = torch::empty({l_agents}, float_opts);
    step_term_gpu_ = torch::empty({l_agents}, float_opts);
    step_trunc_gpu_ = torch::empty({l_agents}, float_opts);
    step_dones_gpu_ = torch::empty({l_agents}, float_opts);

    const int64_t o_agents = t_agents - l_agents;
    opp_obs_gpu_ = torch::empty({o_agents, static_cast<int64_t>(obs_size_)}, float_opts);
    opp_masks_gpu_ = torch::empty({o_agents, static_cast<int64_t>(action_space_size)}, float_opts);
}

void PPORunner::setup_rollout_groups(const torch::Tensor& obs, const torch::Tensor& masks) {
    rollout_groups_.clear();
    for (const env::EnvGroup& env_group : vec_env_->get_groups()) {
        auto group = std::make_unique<RolloutGroup>();
        const auto begin = static_cast<int64_t>(env_group.learner_begin);
        const auto count = static_cast<int64_t>(env_group.learner_count());
        group->learner_begin = begin;
        group->learner_count = count;

        group->obs_src = obs.narrow(0, begin, count);
        group->masks_src = masks.narrow(0, begin, count);
        group->actions_cpu = actions_cpu_.narrow(0, begin, count);
        group->obs_gpu = step_obs_gpu_.narrow(0, begin, count);
        group->masks_gpu = step_masks_gpu_.narrow(0, begin, count);
        group->rewards_gpu = step_rewards_gpu_.narrow(0, begin, count);
        group->term_gpu = step_term_gpu_.narrow(0, begin, count);
        group->trunc_gpu = step_trunc_gpu_.narrow(0, begin, count);
        group->dones_gpu = step_dones_gpu_.narrow(0, begin, count);

        for (const env::RowRange& rows : env_group.opponent_slots) {
            const auto opp_begin = static_cast<int64_t>(rows.begin);
            const auto opp_count = static_cast<int64_t>(rows.count);
            const int64_t staging_begin = opp_begin - static_cast<int64_t>(learner_agents_);
            group->opponents.push_back({
                obs.narrow(0, opp_begin, opp_count),
                masks.narrow(0, opp_begin, opp_count),
                opp_obs_gpu_.narrow(0, staging_begin, opp_count),
                opp_masks_gpu_.narrow(0, staging_begin, opp_count),
                actions_cpu_.narrow(0, opp_begin, opp_count)
            });
        }

        rollout_groups_.push_back(std::move(group));
    }
}

void PPORunner::load_opponent_slot(size_t slot, const std::vector<torch::Tensor>& params) {
    const std::vector<torch::Tensor> slot_params = opponent_slots_.at(slot)->parameters();
    TORCH_CHECK(slot_params.size() == params.size(), "Opponent weights don't match the network config.");

    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < params.size(); i++) {
        if (!params[i].defined()) continue;
        TORCH_CHECK(slot_params[i].sizes() == params[i].sizes(), "Opponent weights don't match the network config.");
        slot_params[i].copy_(params[i], true);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PPORunner::inference_graph_logic(RolloutGroup& group) {
    // The opponent slots fork off here. An opponent forward is a chain of small kernels that leaves the GPU mostly
    // idle, so running each slot on its own stream overlaps it with the learner's forward instead of queuing after it.
    const at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    at::cuda::CUDAEvent fork;
    fork.record(main_stream);

    group.obs_gpu.copy_(group.obs_src, true);
    group.masks_gpu.copy_(group.masks_src, true);

    auto [actions, log_probs, values] = actor_critic_->get_action_and_value(group.obs_gpu, group.masks_gpu);

    // Non-blocking, since synchronizing isn't allowed during capture. `infer_group` syncs after replay.
    group.actions_cpu.copy_(actions.to(torch::kInt32), true);

    for (size_t slot = 0; slot < group.opponents.size(); slot++) {
        RolloutGroup::OpponentRows& opp = group.opponents[slot];
        if (opp.obs_src.size(0) == 0) continue;

        const at::cuda::CUDAStream slot_stream = opponent_streams_[slot];
        fork.block(slot_stream);
        {
            at::cuda::CUDAStreamGuard stream_guard(slot_stream);
            opp.obs_gpu.copy_(opp.obs_src, true);
            opp.masks_gpu.copy_(opp.masks_src, true);
            const torch::Tensor opp_actions = opponent_slots_[slot]->sample_action(opp.obs_gpu, opp.masks_gpu);
            opp.actions_cpu.copy_(opp_actions.to(torch::kInt32), true);
        }

        // Join back. The learner's work is already queued ahead of this wait, so it doesn't wait on the slot.
        // A capture must rejoin every stream it forked before it ends.
        at::cuda::CUDAEvent join;
        join.record(slot_stream);
        join.block(main_stream);
    }

    return { actions, log_probs, values.squeeze(-1) };
}

void PPORunner::infer_group(RolloutGroup& group, const torch::Tensor& obs, const torch::Tensor& masks) {
    if (!group.captured) {
        // Graph capture has to happen on a private stream, after any work already queued on the current stream.
        at::cuda::CUDAStream capture_stream = at::cuda::getStreamFromPool();
        {
            at::cuda::CUDAEvent ready;
            ready.record(at::cuda::getCurrentCUDAStream());
            ready.block(capture_stream);
        }
        {
            at::cuda::CUDAStreamGuard stream_guard(capture_stream);

            // Warmup runs lazy initialization (e.g. cuBLAS workspaces) outside of capture. It only consumes RNG.
            for (int i = 0; i < 3; i++) {
                (void)inference_graph_logic(group);
            }
            capture_stream.synchronize();

            group.graph.capture_begin();
            std::tie(group.actions_gpu, group.log_probs_gpu, group.values_gpu) = inference_graph_logic(group);
            group.graph.capture_end();
        }
        group.captured = true;
    }

    // The graph baked in the source rows' addresses.
    TORCH_CHECK(obs.data_ptr() == group.obs_src.data_ptr() && masks.data_ptr() == group.masks_src.data_ptr(),
                "Rollout inference graph inputs must be the group's rows of VecEnv's persistent obs and mask buffers.");

    group.graph.replay();
    at::cuda::getCurrentCUDAStream().synchronize();
}

void PPORunner::finish_group_step(RolloutGroup& group, size_t step, const env::BatchedStepResult& step_res) {
    group.rewards_gpu.copy_(step_res.rewards, true);
    group.term_gpu.copy_(step_res.terminated, true);
    group.trunc_gpu.copy_(step_res.truncated, true);

    // Store truncated states for deferred bootstrapping.
    const float* truncated = step_res.truncated.data_ptr<float>();
    const float* terminal_obs = step_res.terminal_observations.data_ptr<float>();
    const int64_t obs_size = step_res.terminal_observations.size(1);

    for (int64_t i = 0; i < group.learner_count; i++) {
        if (truncated[i] == 0.0f) continue;
        deferred_idx_.push_back(static_cast<int64_t>(step) * static_cast<int64_t>(learner_agents_) + group.learner_begin + i);
        const float* row = terminal_obs + i * obs_size;
        deferred_obs_.insert(deferred_obs_.end(), row, row + obs_size);
    }

    torch::max_out(group.dones_gpu, group.term_gpu, group.trunc_gpu);

    buffer_->insert(
        step, group.learner_begin,
        group.obs_gpu, group.actions_gpu, group.masks_gpu,
        group.rewards_gpu, group.dones_gpu, group.log_probs_gpu, group.values_gpu
    );
}

void PPORunner::save_checkpoint(const std::string& dir) const {
    const fs::path final_dir = fs::path(dir) / (config_.bot_name + "_" + std::to_string(global_step_));
    const fs::path tmp_dir = final_dir.string() + ".tmp";

    // Write into a temp dir and rename once complete so an interrupted save never looks like a valid checkpoint.
    fs::remove_all(tmp_dir);
    fs::create_directories(tmp_dir);
    torch::save(actor_critic_, (tmp_dir / MODEL_FILE).string());
    trainer_->save_optimizer((tmp_dir / OPTIMIZER_FILE).string());

    fs::remove_all(final_dir);
    fs::rename(tmp_dir, final_dir);

    // The pool saves its snapshots as they're added, so only the latest win records are left to write.
    if (opponent_pool_) {
        opponent_pool_->save_state();
    }
    std::cout << "Checkpoint saved to: " << final_dir.string() << "\n";
}

void PPORunner::write_log_window() {
    LogWindow& window = log_window_;
    const double steps = static_cast<double>(window.steps);
    const double updates = static_cast<double>(window.updates);

    // Rewards and metrics accumulated inside the envs since the last log.
    vec_env_->update_telemetry_breakdowns();
    // Components are the raw values, before the reward options. Magnitudes are the absolute values the agents received.
    double total_raw = 0.0;
    double total_magnitude = 0.0;
    for (const auto& [name, accumulation] : vec_env_->get_reward_breakdown()) {
        total_raw += accumulation.raw;
        total_magnitude += accumulation.magnitude;
        logger_->add_scalar("Reward_Components/" + name, global_step_, static_cast<float>(accumulation.raw / steps));
        logger_->add_scalar("Reward_Magnitude/" + name, global_step_, static_cast<float>(accumulation.magnitude / steps));
    }
    logger_->add_scalar("Reward/Mean_Step", global_step_, static_cast<float>(total_raw / steps));
    logger_->add_scalar("Reward/Mean_Abs_Step", global_step_, static_cast<float>(total_magnitude / steps));

    // Each reward's fraction of the total magnitude, which sums to 1. Skipped when no reward was received.
    if (total_magnitude > 0.0) {
        for (const auto& [name, accumulation] : vec_env_->get_reward_breakdown()) {
            logger_->add_scalar("Reward_Share/" + name, global_step_, static_cast<float>(accumulation.magnitude / total_magnitude));
        }
    }

    // Custom Metrics.
    for (const auto& [name, accumulation] : vec_env_->get_metric_breakdown()) {
        if (accumulation.count == 0) continue;
        logger_->add_scalar("Metrics/" + name, global_step_, static_cast<float>(accumulation.value / accumulation.count));
    }

    logger_->add_scalar("Episode/Goals", global_step_, static_cast<float>(window.goals));
    logger_->add_scalar("Episode/Truncations", global_step_,
                        static_cast<float>(window.no_touch_truncations + window.time_limit_truncations));
    logger_->add_scalar("Episode/Truncations_No_Touch", global_step_, static_cast<float>(window.no_touch_truncations));
    logger_->add_scalar("Episode/Truncations_Time_Limit", global_step_, static_cast<float>(window.time_limit_truncations));

    // PFSP totals over the window. Truncated episodes aren't counted.
    if (!opponent_slots_.empty()) {
        logger_->add_scalar("PFSP/Pool_Size", global_step_, static_cast<float>(opponent_pool_->size()));
        logger_->add_scalar("PFSP/Games", global_step_, static_cast<float>(window.pfsp_games));
        if (window.pfsp_games > 0) {
            logger_->add_scalar("PFSP/Win_Rate", global_step_,
                                static_cast<float>(window.pfsp_wins) / static_cast<float>(window.pfsp_games));
        }
    }

    // Trainer metrics are averaged over the window's updates, which all have the same number of steps.
    const auto mean_metric = [&](const std::string& name) {
        return static_cast<float>(window.train_metrics.at(name) / updates);
    };
    logger_->add_scalar("Performance/Total_SPS", global_step_, steps / window.update_seconds);
    logger_->add_scalar("Performance/Rollout_SPS", global_step_, steps / window.rollout_seconds);
    logger_->add_scalar("Loss/Policy", global_step_, mean_metric("policy_loss"));
    logger_->add_scalar("Loss/Value", global_step_, mean_metric("value_loss"));
    logger_->add_scalar("Loss/Entropy", global_step_, mean_metric("entropy"));
    logger_->add_scalar("Policy/Normalized_Entropy", global_step_, mean_metric("normalized_entropy"));
    logger_->add_scalar("Policy/Approx_KL", global_step_, mean_metric("approx_kl"));
    logger_->add_scalar("Policy/Clip_Fraction", global_step_, mean_metric("clip_fraction"));
    logger_->add_scalar("Value/Explained_Variance", global_step_, mean_metric("explained_variance"));

    window = {};
}

void PPORunner::run_training(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir) {
    
    if (vec_env_->get_groups().size() < 2) {
        throw std::invalid_argument("Training needs at least 2 envs, since the rollout is pipelined over two env groups.");
    }

    // Holds VecEnv's full batched buffers, which always contain every agent's latest obs and masks.
    const auto reset_res = vec_env_->reset();
    setup_rollout_groups(reset_res.observations, reset_res.action_masks);

    // Seed an empty pool with the starting learner, so there is always a version to play.
    const int64_t snapshot_interval = config_.pfsp_cfg.snapshot_interval;
    if (opponent_pool_) {
        if (opponent_pool_->empty()) {
            opponent_pool_->add(global_step_, actor_critic_);
        }
        next_snapshot_step_ = (global_step_ / snapshot_interval + 1) * snapshot_interval;
    }

    // Periodic checkpoints land on multiples of the interval, so they stay on schedule across resumes.
    const int64_t checkpoint_interval = config_.steps_per_checkpoint;
    if (checkpoint_interval > 0) {
        next_checkpoint_step_ = (global_step_ / checkpoint_interval + 1) * checkpoint_interval;
    }

    // TensorBoard logs land on multiples of the interval like checkpoints. 0 logs every update.
    const int64_t log_interval = config_.steps_per_log > 0 ? config_.steps_per_log : total_steps_per_update_;
    next_log_step_ = (global_step_ / log_interval + 1) * log_interval;

    for (int update = 1; update <= num_updates; ++update) {

        if (stop_flag) {
            std::cout << "\nTraining interrupted by user. Stopping...\n";
            break;
        }

        const auto t_start = std::chrono::high_resolution_clock::now();
        const auto rollout_start = std::chrono::high_resolution_clock::now();

        // This update's opponents: past versions sampled from the pool, or a copy of the current learner
        // until the PFSP start steps.
        std::vector<size_t> opponent_picks;
        std::vector<int64_t> opponent_versions; // Kept for the console, since a snapshot added later invalidates the picks.
        if (opponent_pool_ && global_step_ >= config_.pfsp_cfg.start_steps) {
            opponent_picks = opponent_pool_->sample(opponent_slots_.size());
            for (const size_t pick : opponent_picks) {
                opponent_versions.push_back((*opponent_pool_)[pick].steps);
            }
        }
        for (size_t slot = 0; slot < opponent_slots_.size(); slot++) {
            if (opponent_picks.empty()) {
                load_opponent_slot(slot, actor_critic_->parameters());
            } else {
                load_opponent_slot(slot, (*opponent_pool_)[opponent_picks[slot]].params);
            }
        }

        // Pipelined rollout over two env groups: while one group's physics runs on the workers, the main thread
        // uploads the other group's results and runs its inference. Each env still acts on its own latest obs
        // before it advances, so there is no action delay; only the order of steps across groups changes.
        RolloutGroup& group_a = *rollout_groups_[0];
        RolloutGroup& group_b = *rollout_groups_[1];
        int* const actions_ptr = actions_cpu_.data_ptr<int>();
        const size_t num_steps = buffer_size_;

        infer_group(group_a, group_a.obs_src, group_a.masks_src);
        vec_env_->step_async(0, actions_ptr);
        infer_group(group_b, group_b.obs_src, group_b.masks_src);

        for (size_t t = 0; t < num_steps; ++t) {
            const bool has_next = t + 1 < num_steps;

            const auto res_a = vec_env_->wait();
            vec_env_->step_async(1, actions_ptr);
            finish_group_step(group_a, t, res_a);
            if (has_next) infer_group(group_a, res_a.observations, res_a.action_masks);

            const auto res_b = vec_env_->wait();
            if (has_next) vec_env_->step_async(0, actions_ptr);
            finish_group_step(group_b, t, res_b);
            if (has_next) infer_group(group_b, res_b.observations, res_b.action_masks);
        }

        // Successful rollout, increment global steps.
        global_step_ += total_steps_per_update_;

        // Episode counts are collected every update for the console, and summed into the log window.
        const env::EpisodeCounts episodes = vec_env_->collect_episode_counts();
        log_window_.goals += episodes.goals;
        log_window_.no_touch_truncations += episodes.no_touch_truncations;
        log_window_.time_limit_truncations += episodes.time_limit_truncations;

        // PFSP results against this update's opponent slots. Truncated episodes aren't counted.
        uint32_t pfsp_wins = 0;
        uint32_t pfsp_games = 0;
        const auto pfsp_outcomes = vec_env_->collect_pfsp_outcomes();
        for (size_t slot = 0; slot < pfsp_outcomes.size(); slot++) {
            const uint32_t games = pfsp_outcomes[slot].wins + pfsp_outcomes[slot].losses;
            pfsp_wins += pfsp_outcomes[slot].wins;
            pfsp_games += games;
            if (!opponent_picks.empty()) {
                opponent_pool_->record(opponent_picks[slot], pfsp_outcomes[slot].wins, pfsp_outcomes[slot].losses);
            }
        }
        log_window_.pfsp_wins += pfsp_wins;
        log_window_.pfsp_games += pfsp_games;

        // Deferred truncations forward pass for values.
        const int64_t n = static_cast<int64_t>(deferred_idx_.size());
        if (n > 0) {
            torch::Tensor deferred_idx_gpu = torch::from_blob(deferred_idx_.data(), n, torch::kInt64).to(device_);
            torch::Tensor deferred_obs_gpu = torch::from_blob(deferred_obs_.data(), {n, static_cast<int64_t>(obs_size_)}, torch::kFloat32).to(device_);
            torch::Tensor truncated_values;
            {
                torch::NoGradGuard no_grad;
                truncated_values = actor_critic_->forward_critic(deferred_obs_gpu).squeeze(-1);
            }
            
            buffer_->rewards_.view({-1}).index_add_(0, deferred_idx_gpu, truncated_values * config_.ppo_cfg.gae_gamma);
        }
        deferred_idx_.clear();
        deferred_obs_.clear();
        
        const auto rollout_end = std::chrono::high_resolution_clock::now();

        // GAE & Optimize. The full obs buffer holds every group's latest obs, and its first rows are the learners'.
        step_obs_gpu_.copy_(reset_res.observations.narrow(0, 0, static_cast<int64_t>(learner_agents_)), true);
        torch::Tensor next_values;
        {
            torch::NoGradGuard no_grad;
            next_values = actor_critic_->forward_critic(step_obs_gpu_).squeeze(-1);
        }

        const auto gae_start = std::chrono::high_resolution_clock::now();
        buffer_->compute_returns_and_advantages(next_values, config_.ppo_cfg.gae_gamma, config_.ppo_cfg.gae_lambda);
        const auto gae_end = std::chrono::high_resolution_clock::now();
        const auto metrics = trainer_->train_step(*buffer_);

        // Snapshot the freshly trained learner into the pool. Invalidates `opponent_picks`.
        if (opponent_pool_ && global_step_ >= next_snapshot_step_) {
            opponent_pool_->add(global_step_, actor_critic_);
            next_snapshot_step_ = (global_step_ / snapshot_interval + 1) * snapshot_interval;
        }

        const auto t_end = std::chrono::high_resolution_clock::now();
        
        const std::chrono::duration<double> update_time = t_end - t_start;
        const std::chrono::duration<double> rollout_time = rollout_end - rollout_start;
        const std::chrono::duration<double> gae_time = gae_end - gae_start;
        const double train_time = update_time.count() - rollout_time.count() - gae_time.count();

        const double total_sps = total_steps_per_update_ / update_time.count();
        const double rollout_sps = total_steps_per_update_ / rollout_time.count();

        log_window_.steps += total_steps_per_update_;
        log_window_.updates++;
        log_window_.update_seconds += update_time.count();
        log_window_.rollout_seconds += rollout_time.count();
        for (const auto& [name, value] : metrics) {
            log_window_.train_metrics[name] += value;
        }
        if (global_step_ >= next_log_step_) {
            write_log_window();
            next_log_step_ = (global_step_ / log_interval + 1) * log_interval;
        }

        std::cout << "Update: " << update 
                  << "\nLifetime Steps: " << global_step_
                  << "\n | Total SPS:   " << static_cast<int64_t>(total_sps)
                  << "\n |  | Rollout:  " << static_cast<int64_t>(rollout_sps)
                  << "\n | Time:        " << std::fixed << std::setprecision(4) << update_time.count() << "s"
                  << "\n |  | Rollout:  " << rollout_time.count() << "s"
                  << "\n |  | GAE:      " << gae_time.count() << "s"
                  << "\n |  | Train:    " << train_time << "s"
                  << "\n | Policy Loss: " << std::defaultfloat << std::setprecision(6) << metrics.at("policy_loss")
                  << "\n | Value Loss:  " << metrics.at("value_loss")
                  << "\n | Entropy:     " << metrics.at("entropy")
                  << "\n | Norm Ent:    " << metrics.at("normalized_entropy")
                  << "\n | Approx KL:   " << metrics.at("approx_kl")
                  << "\n | Clip Frac:   " << metrics.at("clip_fraction")
                  << "\n | Expl Var:    " << metrics.at("explained_variance")
                  << "\n | Episodes:    " << episodes.goals << " goals, " << episodes.no_touch_truncations << " no touch, "
                  << episodes.time_limit_truncations << " time limit\n";
        if (!pfsp_outcomes.empty()) {
            std::cout << " | PFSP Wins:   " << pfsp_wins << " / " << pfsp_games << " games";
            if (opponent_versions.empty()) {
                std::cout << " vs current learner";
            } else {
                std::cout << " vs";
                for (size_t slot = 0; slot < opponent_versions.size(); slot++) {
                    std::cout << (slot == 0 ? " " : ", ") << opponent_versions[slot];
                }
            }
            std::cout << " (pool: " << opponent_pool_->size() << ")\n";
        }

        if (checkpoint_interval > 0 && global_step_ >= next_checkpoint_step_) {
            save_checkpoint(checkpoint_dir);
            last_checkpoint_step_ = global_step_;
            next_checkpoint_step_ = (global_step_ / checkpoint_interval + 1) * checkpoint_interval;
        }
    }

    // Log the partial window left by an interrupted or finished run.
    if (log_window_.updates > 0) {
        write_log_window();
    }

    // Skip the final save if the last update already wrote this checkpoint.
    if (global_step_ != last_checkpoint_step_) {
        save_checkpoint(checkpoint_dir);
    }
}

void PPORunner::run_render(const std::atomic<bool>& stop_flag) {
    auto reset_res = vec_env_->reset();
    torch::Tensor current_obs = reset_res.observations;
    torch::Tensor current_masks = reset_res.action_masks;

    std::cout << "Starting visualizer... Press Ctrl+C to stop.\n";

    while (!stop_flag) {
        step_obs_gpu_.copy_(current_obs, true);
        step_masks_gpu_.copy_(current_masks, true);

        auto [actions_gpu, log_probs_gpu, values_gpu] = actor_critic_->get_action_and_value(step_obs_gpu_, step_masks_gpu_);

        actions_cpu_.copy_(actions_gpu, false);
        
        // VecEnv blocks to maintain regular time scale.
        auto step_res = vec_env_->step(actions_cpu_.data_ptr<int>()); 

        current_obs = step_res.observations;
        current_masks = step_res.action_masks;
    }
}

void PPORunner::run(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir) {
    global_step_ = load_latest_checkpoint(checkpoint_dir);

    if (opponent_pool_) {
        opponent_pool_->open(fs::path(checkpoint_dir) / (config_.bot_name + POOL_DIR_SUFFIX), global_step_);
    }

    if (config_.render) {
        run_render(stop_flag);
    } else {
        run_training(num_updates, stop_flag, checkpoint_dir);
    }
}

} // namespace buta_ppo::rl