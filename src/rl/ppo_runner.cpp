// buta-ppo/src/rl/ppo_runner.hpp
#include "ppo_runner.hpp"
#include "tensorboard_logger.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

namespace buta_ppo::rl {

PPORunner::PPORunner(const RunnerConfig& config)
    : config_(config), device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
    
    if (torch::cuda::is_available()) {
        std::cout << "CUDA detected. Running on GPU." << std::endl;
        at::globalContext().setUserEnabledCuDNN(true);
        at::globalContext().setBenchmarkCuDNN(true);
    } else {
        std::cout << "CUDA not found. Defaulting to CPU." << std::endl;
    }

    std::string log_file = "logs/" + config_.bot_name + ".tfevents";
    logger_ = std::make_unique<TensorBoardLogger>(log_file.c_str());

    setup_dimensions_and_buffers();
}

int64_t PPORunner::load_latest_checkpoint(const std::string& dir) {
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
        return 0;
    }

    std::string latest_file;
    int64_t max_steps = -1;

    // Scan directory for .pt files
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".pt") {
            std::string filename = entry.path().stem().string();
            
            size_t delim_pos = filename.find_last_of('_');
            if (delim_pos != std::string::npos) {
                try {
                    // Extract the string after the last '_' and convert to int64
                    int64_t steps = std::stoll(filename.substr(delim_pos + 1));
                    
                    // Only load if the name prefix matches the config
                    std::string prefix = filename.substr(0, delim_pos);
                    if (prefix == config_.bot_name && steps > max_steps) {
                        max_steps = steps;
                        latest_file = entry.path().string();
                    }
                } catch (const std::exception&) {
                    // Ignore any other files.
                }
            }
        }
    }

    if (max_steps >= 0 && !latest_file.empty()) {
        torch::load(actor_critic_, latest_file);
        std::cout << "Latest Model Found: " << latest_file << " (Lifetime Steps: " << max_steps << ")\n";
        return max_steps;
    }
    
    std::cout << "No valid checkpoints found for bot '" << config_.bot_name << "'. Starting fresh training.\n";
    return 0;
}

void PPORunner::setup_dimensions_and_buffers() {

    if (config_.render) {
        config_.num_envs = 1;
        config_.num_minibatches = 1;
    }

    vec_env_ = std::make_unique<env::VecEnv>(
        config_.num_envs, 
        config_.match_distributions,
        std::min(config_.num_envs, config_.num_threads), 
        config_.ticks_per_step, 
        config_.max_players_per_team,
        config_.render
    );

    total_agents_ = vec_env_->get_total_agents();
    size_t obs_size = vec_env_->get_single_obs_size();
    size_t action_space_size = vec_env_->get_action_space_size();

    if (config_.render) {
        config_.target_steps_per_update = total_agents_;
    }

    buffer_size_ = (config_.target_steps_per_update + total_agents_ - 1) / total_agents_;
    while ((buffer_size_ * total_agents_) % config_.num_minibatches != 0) {
        buffer_size_++;
    }

    total_steps_per_update_ = static_cast<int64_t>(buffer_size_ * total_agents_);
    config_.ppo_cfg.mini_batch_size = total_steps_per_update_ / config_.num_minibatches;

    config_.ac_cfg.obs_size = obs_size;
    config_.ac_cfg.action_size = action_space_size;
    
    actor_critic_ = ActorCritic(config_.ac_cfg);
    actor_critic_->to(device_);

    trainer_ = std::make_unique<PPOTrainer>(config_.ppo_cfg, actor_critic_, device_);
    
    buffer_ = std::make_unique<RolloutBuffer>(
        buffer_size_, total_agents_, obs_size, action_space_size, device_
    );

    auto cpu_int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    actions_cpu_ = torch::empty({(int64_t)total_agents_}, cpu_int_opts);

    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    step_obs_gpu_ = torch::empty({(int64_t)total_agents_, (int64_t)obs_size}, float_opts);
    step_masks_gpu_ = torch::empty({(int64_t)total_agents_, (int64_t)action_space_size}, float_opts);
    step_rewards_gpu_ = torch::empty({(int64_t)total_agents_}, float_opts);
    step_dones_gpu_ = torch::empty({(int64_t)total_agents_}, float_opts);
}

void PPORunner::save_checkpoint(const std::string& dir) const {
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
    
    std::string path = dir + "/" + config_.bot_name + "_" + std::to_string(global_step_) + ".pt";
    torch::save(actor_critic_, path);
    std::cout << "Model checkpoint saved to: " << path << std::endl;
}

void PPORunner::run_training(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir) {
    
    auto reset_res = vec_env_->reset();

    for (int update = 1; update <= num_updates; ++update) {

        if (stop_flag) {
            std::cout << "\nTraining interrupted by user. Stopping..." << std::endl;
            break;
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        buffer_->reset();

        torch::Tensor current_obs = reset_res.observations;
        torch::Tensor current_masks = reset_res.action_masks;

        auto rollout_start = std::chrono::high_resolution_clock::now();
        
        while (!buffer_->is_full()) {
            step_obs_gpu_.copy_(current_obs, true);
            step_masks_gpu_.copy_(current_masks, true);

            auto [actions_gpu, log_probs_gpu, values_gpu] = actor_critic_->get_action_and_value(step_obs_gpu_, step_masks_gpu_);

            actions_cpu_.copy_(actions_gpu, false); // Blocking sync required for physics
            
            auto step_res = vec_env_->step(actions_cpu_.data_ptr<int>());

            step_rewards_gpu_.copy_(step_res.rewards, true);
            step_dones_gpu_.copy_(step_res.dones, true);

            buffer_->insert(
                step_obs_gpu_, actions_gpu, step_masks_gpu_, 
                step_rewards_gpu_, step_dones_gpu_, log_probs_gpu, values_gpu.squeeze(-1)
            );

            current_obs = step_res.observations;
            current_masks = step_res.action_masks;
        }

        // Successful rollout, increment global steps.
        global_step_ += total_steps_per_update_;
        float mean_step_reward = buffer_->rewards_.mean().item<float>();
        logger_->add_scalar("Reward/Mean_Step", global_step_, mean_step_reward);

        vec_env_->update_reward_breakdown();
        const auto& reward_breakdown = vec_env_->get_reward_breakdown();

        for (const auto& [name, total_weighted_reward] : reward_breakdown) {
            double avg_per_step = total_weighted_reward / static_cast<double>(total_steps_per_update_);
            logger_->add_scalar("Reward_Components/" + name, global_step_, static_cast<float>(avg_per_step));
        }
        
        auto rollout_end = std::chrono::high_resolution_clock::now();

        // GAE & Optimize
        step_obs_gpu_.copy_(current_obs, true);
        torch::Tensor next_values;
        {
            torch::NoGradGuard no_grad;
            auto [logits, values] = actor_critic_->forward(step_obs_gpu_);
            next_values = values.squeeze(-1);
        }

        buffer_->compute_returns_and_advantages(next_values, step_dones_gpu_);
        auto metrics = trainer_->train_step(*buffer_);

        auto t_end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> update_time = t_end - t_start;
        std::chrono::duration<double> rollout_time = rollout_end - rollout_start;
        double train_time = update_time.count() - rollout_time.count();

        double total_sps = total_steps_per_update_ / update_time.count();
        double rollout_sps = total_steps_per_update_ / rollout_time.count();

        logger_->add_scalar("Performance/Total_SPS", global_step_, total_sps);
        logger_->add_scalar("Performance/Rollout_SPS", global_step_, rollout_sps);
        logger_->add_scalar("Loss/Policy", global_step_, metrics["policy_loss"]);
        logger_->add_scalar("Loss/Value", global_step_, metrics["value_loss"]);
        logger_->add_scalar("Loss/Entropy", global_step_, metrics["entropy"]);

        std::cout << "Update: " << update 
                  << "\nLifetime Steps: " << global_step_
                  << "\n | Total SPS:   " << static_cast<int64_t>(total_sps)
                  << "\n |  | Rollout:  " << static_cast<int64_t>(rollout_sps)
                  << "\n | Time:        " << std::fixed << std::setprecision(2) << update_time.count() << "s"
                  << "\n |  | Rollout:  " << rollout_time.count() << "s"
                  << "\n |  | Train:    " << train_time << "s"
                  << "\n | Policy Loss: " << std::defaultfloat << std::setprecision(6) << metrics["policy_loss"]
                  << "\n | Value Loss:  " << metrics["value_loss"]
                  << "\n | Entropy:     " << metrics["entropy"] << std::endl;
    }

    save_checkpoint(checkpoint_dir);
}

void PPORunner::run_render(const std::atomic<bool>& stop_flag) {
    auto reset_res = vec_env_->reset();
    torch::Tensor current_obs = reset_res.observations;
    torch::Tensor current_masks = reset_res.action_masks;

    std::cout << "Starting visualizer... Press Ctrl+C to stop." << std::endl;

    while (!stop_flag) {
        step_obs_gpu_.copy_(current_obs, true);
        step_masks_gpu_.copy_(current_masks, true);

        auto [actions_gpu, log_probs_gpu, values_gpu] = actor_critic_->get_action_and_value(step_obs_gpu_, step_masks_gpu_);

        actions_cpu_.copy_(actions_gpu, false);
        
        // vec_env_->step() will now naturally block for ~66ms while rendering smoothly
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

}; // namespace buta_ppo::rl