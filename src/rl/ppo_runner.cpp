// buta-ppo/src/rl/ppo_runner.cpp
#include "ppo_runner.hpp"
#include "rollout_buffer.hpp"
#include "tensorboard_logger.h"
#include "env/vec_env.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>

#include <charconv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <memory>
#include <sstream> // [PERF-INSTRUMENTATION]
#include <stdexcept>

namespace fs = std::filesystem;

namespace buta_ppo::rl {

namespace {
// File names inside each `<bot name>_<steps>/` checkpoint directory.
constexpr const char* MODEL_FILE = "model.pt";
constexpr const char* OPTIMIZER_FILE = "optimizer.pt";
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
        config_.num_threads, 
        config_.ticks_per_step, 
        config_.max_players_per_team,
        config_.render
    );

    total_agents_ = vec_env_->get_total_agents();
    const size_t obs_size = vec_env_->get_single_obs_size();
    const size_t action_space_size = vec_env_->get_action_space_size();

    if (config_.render) {
        config_.target_steps_per_update = total_agents_;
    }

    buffer_size_ = (config_.target_steps_per_update + total_agents_ - 1) / total_agents_;
    while ((buffer_size_ * total_agents_) % config_.num_minibatches != 0) {
        buffer_size_++;
    }

    total_steps_per_update_ = static_cast<int64_t>(buffer_size_ * total_agents_);
    config_.ppo_cfg.mini_batch_size = total_steps_per_update_ / config_.num_minibatches;

    config_.ac_cfg.obs_size = static_cast<int64_t>(obs_size);
    config_.ac_cfg.action_size = static_cast<int64_t>(action_space_size);
    
    actor_critic_ = ActorCritic(config_.ac_cfg);
    actor_critic_->to(device_);

    trainer_ = std::make_unique<PPOTrainer>(config_.ppo_cfg, actor_critic_, device_);
    
    buffer_ = std::make_unique<RolloutBuffer>(
        buffer_size_, total_agents_, obs_size, action_space_size, device_
    );

    const int64_t t_agents = static_cast<int64_t>(total_agents_);
    const auto cpu_int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    actions_cpu_ = torch::empty({t_agents}, cpu_int_opts);

    const auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    step_obs_gpu_ = torch::empty({t_agents, static_cast<int64_t>(obs_size)}, float_opts);
    step_masks_gpu_ = torch::empty({t_agents, static_cast<int64_t>(action_space_size)}, float_opts);
    step_rewards_gpu_ = torch::empty({t_agents}, float_opts);
    step_term_gpu_ = torch::empty({t_agents}, float_opts);
    step_trunc_gpu_ = torch::empty({t_agents}, float_opts);
    step_terminal_obs_gpu_ = torch::empty({t_agents, static_cast<int64_t>(obs_size)}, float_opts);
    step_dones_gpu_ = torch::empty({t_agents}, float_opts);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PPORunner::inference_graph_logic() {
    step_obs_gpu_.copy_(graph_obs_src_, true);
    step_masks_gpu_.copy_(graph_masks_src_, true);

    auto [actions, log_probs, values] = actor_critic_->get_action_and_value(step_obs_gpu_, step_masks_gpu_);

    // Non-blocking, since synchronizing isn't allowed during capture. `infer_rollout_actions` syncs after replay.
    actions_cpu_.copy_(actions.to(torch::kInt32), true);

    return { actions, log_probs, values.squeeze(-1) };
}

void PPORunner::infer_rollout_actions(const torch::Tensor& obs, const torch::Tensor& masks) {
    if (!inference_graph_captured_) {
        graph_obs_src_ = obs;
        graph_masks_src_ = masks;

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
                (void)inference_graph_logic();
            }
            capture_stream.synchronize();

            inference_graph_.capture_begin();
            std::tie(graph_actions_gpu_, graph_log_probs_gpu_, graph_values_gpu_) = inference_graph_logic();
            inference_graph_.capture_end();
        }
        inference_graph_captured_ = true;
    }

    // The graph baked in the source buffers' addresses.
    TORCH_CHECK(obs.data_ptr() == graph_obs_src_.data_ptr() && masks.data_ptr() == graph_masks_src_.data_ptr(),
                "Rollout inference graph inputs must be VecEnv's persistent obs and mask buffers.");

    inference_graph_.replay();
    at::cuda::getCurrentCUDAStream().synchronize();
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
    std::cout << "Checkpoint saved to: " << final_dir.string() << "\n";
}

void PPORunner::run_training(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir) {
    
    auto reset_res = vec_env_->reset();

    for (int update = 1; update <= num_updates; ++update) {

        if (stop_flag) {
            std::cout << "\nTraining interrupted by user. Stopping...\n";
            break;
        }

        const auto t_start = std::chrono::high_resolution_clock::now();
        buffer_->reset();

        torch::Tensor current_obs = reset_res.observations;
        torch::Tensor current_masks = reset_res.action_masks;

        const auto rollout_start = std::chrono::high_resolution_clock::now();

        // [PERF-INSTRUMENTATION] Temporary per-step timing split. Remove after profiling.
        double perf_gpu_s = 0.0;
        double perf_env_s = 0.0;
        int64_t perf_steps = 0;
        
        while (!buffer_->is_full()) {
            const auto perf_s0 = std::chrono::steady_clock::now(); // [PERF-INSTRUMENTATION]
            // Blocks until the actions are on the host, as required for physics.
            infer_rollout_actions(current_obs, current_masks);
            
            const auto perf_s1 = std::chrono::steady_clock::now(); // [PERF-INSTRUMENTATION]
            auto step_res = vec_env_->step(actions_cpu_.data_ptr<int>());
            const auto perf_s2 = std::chrono::steady_clock::now(); // [PERF-INSTRUMENTATION]
            perf_gpu_s += std::chrono::duration<double>(perf_s1 - perf_s0).count();
            perf_env_s += std::chrono::duration<double>(perf_s2 - perf_s1).count();
            perf_steps++;

            step_rewards_gpu_.copy_(step_res.rewards, true);

            step_term_gpu_.copy_(step_res.terminated, true);
            step_trunc_gpu_.copy_(step_res.truncated, true);

            // Bootstrap any truncated states.
            if (step_res.truncated.any().item<bool>()) {
                step_terminal_obs_gpu_.copy_(step_res.terminal_observations, true);

                torch::Tensor bootstrap_values;
                {
                    torch::NoGradGuard no_grad;
                    auto values = actor_critic_->forward_critic(step_terminal_obs_gpu_);
                    bootstrap_values = values.squeeze(-1);
                }

                // This is step_rewards_gpu_ += bootstrap_values * step_trunc_gpu_ * config_.ppo_cfg.gae_gamma.
                step_rewards_gpu_.add_(bootstrap_values * step_trunc_gpu_, config_.ppo_cfg.gae_gamma);
            }

            torch::max_out(step_dones_gpu_, step_term_gpu_, step_trunc_gpu_);

            buffer_->insert(
                step_obs_gpu_, graph_actions_gpu_, step_masks_gpu_, 
                step_rewards_gpu_, step_dones_gpu_, graph_log_probs_gpu_, graph_values_gpu_
            );

            current_obs = step_res.observations;
            current_masks = step_res.action_masks;
        }

        // Successful rollout, increment global steps.
        global_step_ += total_steps_per_update_;

        vec_env_->update_reward_breakdown();
        const auto& reward_breakdown = vec_env_->get_reward_breakdown();

        double total_reward = 0.0;
        
        for (const auto& [name, total_weighted_reward] : reward_breakdown) {
            total_reward += total_weighted_reward;
            const double avg_per_step = total_weighted_reward / static_cast<double>(total_steps_per_update_);
            logger_->add_scalar("Reward_Components/" + name, global_step_, static_cast<float>(avg_per_step));
        }
        logger_->add_scalar("Reward/Mean_Step", global_step_, static_cast<float>(total_reward / static_cast<double>(total_steps_per_update_)));
        
        const auto rollout_end = std::chrono::high_resolution_clock::now();

        // GAE & Optimize.
        step_obs_gpu_.copy_(current_obs, true);
        torch::Tensor next_values;
        {
            torch::NoGradGuard no_grad;
            auto values = actor_critic_->forward_critic(step_obs_gpu_);
            next_values = values.squeeze(-1);
        }

        const auto gae_start = std::chrono::high_resolution_clock::now();
        buffer_->compute_returns_and_advantages(next_values, config_.ppo_cfg.gae_gamma, config_.ppo_cfg.gae_lambda);
        const auto gae_end = std::chrono::high_resolution_clock::now();
        const auto metrics = trainer_->train_step(*buffer_);

        const auto t_end = std::chrono::high_resolution_clock::now();
        
        const std::chrono::duration<double> update_time = t_end - t_start;
        const std::chrono::duration<double> rollout_time = rollout_end - rollout_start;
        const std::chrono::duration<double> gae_time = gae_end - gae_start;
        const double train_time = update_time.count() - rollout_time.count() - gae_time.count();

        const double total_sps = total_steps_per_update_ / update_time.count();
        const double rollout_sps = total_steps_per_update_ / rollout_time.count();

        logger_->add_scalar("Performance/Total_SPS", global_step_, total_sps);
        logger_->add_scalar("Performance/Rollout_SPS", global_step_, rollout_sps);
        logger_->add_scalar("Loss/Policy", global_step_, metrics.at("policy_loss"));
        logger_->add_scalar("Loss/Value", global_step_, metrics.at("value_loss"));
        logger_->add_scalar("Loss/Entropy", global_step_, metrics.at("entropy"));
        logger_->add_scalar("Policy/Approx_KL", global_step_, metrics.at("approx_kl"));
        logger_->add_scalar("Policy/Clip_Fraction", global_step_, metrics.at("clip_fraction"));

        // [PERF-INSTRUMENTATION] Per-step averages in ms.
        const auto [perf_busy_max_s, perf_busy_mean_s] = vec_env_->take_perf_busy();
        const double perf_div = 1e3 / static_cast<double>(std::max<int64_t>(1, perf_steps));
        std::ostringstream perf_report;
        perf_report << std::fixed << std::setprecision(3)
                    << "\n | [PERF] ms/step over " << perf_steps << " steps"
                    << "\n |  | GPU round trip:   " << perf_gpu_s * perf_div
                    << "\n |  | Env wall:         " << perf_env_s * perf_div
                    << "\n |  |  | Busy max:      " << perf_busy_max_s * perf_div
                    << "\n |  |  | Busy mean:     " << perf_busy_mean_s * perf_div
                    << "\n |  |  | Wake overhead: " << (perf_env_s - perf_busy_max_s) * perf_div
                    << "\n |  |  | Imbalance:     " << (perf_busy_max_s - perf_busy_mean_s) * perf_div;

        std::cout << "Update: " << update 
                  << "\nLifetime Steps: " << global_step_
                  << "\n | Total SPS:   " << static_cast<int64_t>(total_sps)
                  << "\n |  | Rollout:  " << static_cast<int64_t>(rollout_sps)
                  << "\n | Time:        " << std::fixed << std::setprecision(4) << update_time.count() << "s"
                  << "\n |  | Rollout:  " << rollout_time.count() << "s"
                  << "\n |  | GAE:      " << gae_time.count() << "s"
                  << "\n |  | Train:    " << train_time << "s"
                  << perf_report.str()
                  << "\n | Policy Loss: " << std::defaultfloat << std::setprecision(6) << metrics.at("policy_loss")
                  << "\n | Value Loss:  " << metrics.at("value_loss")
                  << "\n | Entropy:     " << metrics.at("entropy")
                  << "\n | Approx KL:   " << metrics.at("approx_kl")
                  << "\n | Clip Frac:   " << metrics.at("clip_fraction") << "\n";
    }

    save_checkpoint(checkpoint_dir);
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

    if (config_.render) {
        run_render(stop_flag);
    } else {
        run_training(num_updates, stop_flag, checkpoint_dir);
    }
}

} // namespace buta_ppo::rl