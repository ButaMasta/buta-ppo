#include "buta_ppo/ffi.h"
#include "obs/default_obs.hpp"
#include "action/default_action.hpp"
#include "env/rocketsim_env.hpp"
#include "rl/actor_critic.hpp"

#include <torch/torch.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

extern "C" {

struct BotContext {
    // Read from the model file, so the obs builder and network match what it was trained with.
    buta_ppo::rl::ModelConfig config;
    std::unique_ptr<buta_ppo::obs::DefaultObs> obs_builder;
    std::unique_ptr<buta_ppo::action::DefaultAction> action_parser;
    buta_ppo::rl::ActorCritic model{nullptr};
    // Roster the current slot shuffle was built from.
    std::vector<buta_ppo::env::AgentMeta> agents;
};

void* bot_api_init(const char* checkpoint_path) {
    auto ctx = std::make_unique<BotContext>();

    try {
        const std::optional<buta_ppo::rl::ModelConfig> config = buta_ppo::rl::read_model_config(checkpoint_path);
        if (!config) {
            throw std::runtime_error(std::string(checkpoint_path) + " has no stored model config, since it was saved "
                                     "before models stored one. Re-save it first by running training with its "
                                     "checkpoint as the latest and 0 updates (see docs/deploying.md)");
        }
        ctx->config = *config;

        ctx->obs_builder = std::make_unique<buta_ppo::obs::DefaultObs>(
            static_cast<size_t>(config->max_players_per_team), static_cast<uint8_t>(config->ticks_per_step));
        ctx->action_parser = std::make_unique<buta_ppo::action::DefaultAction>();

        // The model stores its settings but not the obs and action code, which this build compiled in.
        const auto obs_size = static_cast<int64_t>(ctx->obs_builder->get_obs_size());
        const auto action_size = static_cast<int64_t>(ctx->action_parser->get_action_space_size());
        if (obs_size != config->network.obs_size || action_size != config->network.action_size) {
            throw std::runtime_error("the model was trained with obs_size " + std::to_string(config->network.obs_size) +
                                     " and action_size " + std::to_string(config->network.action_size) +
                                     ", but this build's obs builder and action parser give " +
                                     std::to_string(obs_size) + " and " + std::to_string(action_size) +
                                     ". The obs or action code changed since the model was trained");
        }

        // Inference runs on the CPU, on this thread only. A decision takes about 1-2 ms, while LibTorch's default
        // thread pool competes with the game for every core and can stall a decision past the 8.33 ms between game
        // ticks when the CPU is busy.
        torch::set_num_threads(1);
        ctx->model = buta_ppo::rl::ActorCritic(config->network);
        ctx->model->load_weights(checkpoint_path);
    } catch (const std::exception& e) {
        std::cerr << "[buta_ppo_bot] Could not load the model: " << e.what() << std::endl;
        return nullptr;
    }

    std::cout << "[buta_ppo_bot] Loaded " << checkpoint_path << ": "
              << buta_ppo::rl::describe_model_config(ctx->config) << "\n";
    ctx->model->eval();
    return ctx.release();
}

buta_ppo::ffi::CarControls bot_api_get_action(void* ctx_ptr, const buta_ppo::ffi::ArenaState* state, uint32_t car_idx, bool new_kickoff) {
    try {
        auto ctx = static_cast<BotContext*>(ctx_ptr);

        // Rebuild the roster if a player joined/left or changed team.
        bool roster_changed = ctx->agents.size() != state->num_cars;
        for (uint32_t i = 0; !roster_changed && i < state->num_cars; ++i) {
            roster_changed = ctx->agents[i].team != static_cast<buta_ppo::ffi::Team>(state->cars[i].team);
        }
        if (roster_changed) {
            ctx->agents.clear();
            for (uint32_t i = 0; i < state->num_cars; ++i) {
                ctx->agents.push_back({i, static_cast<buta_ppo::ffi::Team>(state->cars[i].team)});
            }
        }

        // Reshuffle slots once per kickoff (the in-game equivalent of reset), or when the roster changes.
        if (new_kickoff || roster_changed) {
            ctx->obs_builder->team_indices_rand(ctx->agents);
        }

        // Build obs.
        const size_t obs_size = ctx->obs_builder->get_obs_size();
        std::vector<float> obs_buffer(obs_size, 0.0f);
        const bool invert_x = ctx->obs_builder->build_obs(*state, ctx->agents, car_idx, obs_buffer.data());

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
        );

        auto mask_tensor = torch::from_blob(
            mask_buffer.data(), 
            {1, static_cast<int64_t>(action_size)}, 
            torch::kFloat32
        );

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
        buta_ppo::ffi::CarControls controls = ctx->action_parser->get_action(static_cast<size_t>(action_idx));
        // Invert necessary controls if the state was X-mirrored.
        if (invert_x) {
            controls.steer  *= -1.0f;
            controls.yaw    *= -1.0f;
            controls.roll   *= -1.0f;
        }
        return controls;

    } catch (const c10::Error& e) {
        std::cerr << "\n[LibTorch Error] " << e.what() << "\n" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\n[C++ Standard Exception] " << e.what() << "\n" << std::endl;
    } catch (...) {
        std::cerr << "\n[Unknown C++ Exception in bot_api_get_action!]\n" << std::endl;
    }

    return buta_ppo::ffi::CarControls{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, false, false, false};
}

uint32_t bot_api_max_players_per_team(void* ctx_ptr) {
    return static_cast<uint32_t>(static_cast<BotContext*>(ctx_ptr)->config.max_players_per_team);
}

uint32_t bot_api_tick_skip(void* ctx_ptr) {
    return static_cast<uint32_t>(static_cast<BotContext*>(ctx_ptr)->config.ticks_per_step);
}

void bot_api_free(void* ctx_ptr) {
    if (ctx_ptr) {
        delete static_cast<BotContext*>(ctx_ptr);
    }
}

} // extern "C"