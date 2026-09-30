#include "buta_ppo/ffi.h"
#include "obs/advanced_obs.hpp"
#include "action/default_action.hpp"
#include "env/rocketsim_env.hpp"
#include "rl/actor_critic.hpp"

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <memory>
#include <vector>
#include <iostream>

extern "C" {

struct BotContext {
    std::unique_ptr<buta_ppo::obs::AdvancedObs> obs_builder;
    std::unique_ptr<buta_ppo::action::DefaultAction> action_parser;
    buta_ppo::rl::ActorCritic model{nullptr};
    torch::Device device{torch::kCPU};
};

void* bot_api_init(const char* checkpoint_path) {
    BotContext* ctx = new BotContext();
    
    ctx->obs_builder = std::make_unique<buta_ppo::obs::AdvancedObs>(3, 8);
    ctx->action_parser = std::make_unique<buta_ppo::action::DefaultAction>();

    buta_ppo::rl::ActorCriticConfig config;
    config.obs_size = static_cast<int64_t>(ctx->obs_builder->get_obs_size());
    config.action_size = static_cast<int64_t>(ctx->action_parser->get_action_space_size());
    config.shared_layers = {512, 512};
    config.actor_layers = {256, 256};
    config.critic_layers = {512, 512};
    config.use_layer_norm = true;
    
    ctx->model = buta_ppo::rl::ActorCritic(config);
    
    if (torch::cuda::is_available()) {
        ctx->device = torch::Device(torch::kCUDA);
        cudaDeviceProp* properties = at::cuda::getDeviceProperties(ctx->device.index());
        std::cout << "[buta_ppo_bot] Device: GPU (CUDA - " 
                << properties->name << ")\n";
    } else {
        ctx->device = torch::Device(torch::kCPU);
        std::cout << "[buta_ppo_bot] Device: CPU\n";
    }
    ctx->model->to(ctx->device);

    try {
        torch::load(ctx->model, checkpoint_path, ctx->device);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading LibTorch model from " << checkpoint_path << ": " << e.what() << std::endl;
        delete ctx;
        return nullptr;
    }

    ctx->model->eval();
    return ctx;
}

buta_ppo::ffi::CarControls bot_api_get_action(void* ctx_ptr, const buta_ppo::ffi::ArenaState* state, uint32_t car_idx) {
    try {
        auto ctx = static_cast<BotContext*>(ctx_ptr);
        
        // Build AgentMeta list from current arena state
        std::vector<buta_ppo::env::AgentMeta> agents;
        agents.reserve(state->num_cars);
        for (uint32_t i = 0; i < state->num_cars; ++i) {
            agents.push_back({i, static_cast<buta_ppo::ffi::Team>(state->cars[i].team)});
        }

        // Build obs.
        const size_t obs_size = ctx->obs_builder->get_obs_size();
        std::vector<float> obs_buffer(obs_size, 0.0f);
        ctx->obs_builder->build_obs(*state, agents, car_idx, obs_buffer.data());

        // Get action mask.
        const size_t action_size = ctx->action_parser->get_action_space_size();
        std::vector<float> mask_buffer(action_size, 1.0f);
        ctx->action_parser->get_action_mask(state->cars[car_idx], mask_buffer.data());

        torch::NoGradGuard no_grad;

        // Setup tensors.
        auto obs_tensor = torch::from_blob(
            obs_buffer.data(), 
            {1, static_cast<int64_t>(obs_size)}, 
            torch::kFloat32
        ).to(ctx->device);

        auto mask_tensor = torch::from_blob(
            mask_buffer.data(), 
            {1, static_cast<int64_t>(action_size)}, 
            torch::kFloat32
        ).to(ctx->device);

        // Evaluate policy logits.
        torch::Tensor logits = ctx->model->forward_actor(obs_tensor);

        // Apply mask.
        torch::Tensor masked_logits = torch::where(
            mask_tensor > 0.5f,
            logits,
            torch::full_like(logits, -1e9f)
        );

        // Select best valid action and decode controls.
        int64_t action_idx = masked_logits.argmax(1).item<int64_t>();
        return ctx->action_parser->get_action(static_cast<size_t>(action_idx));

    } catch (const c10::Error& e) {
        std::cerr << "\n[LibTorch CUDA/Tensor Error] " << e.what() << "\n" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\n[C++ Standard Exception] " << e.what() << "\n" << std::endl;
    } catch (...) {
        std::cerr << "\n[Unknown C++ Exception in bot_api_get_action!]\n" << std::endl;
    }

    return buta_ppo::ffi::CarControls{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, false, false, false};
}

void bot_api_free(void* ctx_ptr) {
    if (ctx_ptr) {
        delete static_cast<BotContext*>(ctx_ptr);
    }
}

} // extern "C"