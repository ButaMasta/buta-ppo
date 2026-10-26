// buta-ppo/src/env/eval_vec_env.cpp
#include "eval_vec_env.hpp"
#include "rocketsim_env.hpp"
#include "episode_limits.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace buta_ppo::env {

EvalVecEnv::EvalVecEnv(
    std::vector<EvalGame> games,
    size_t num_anchor_slots,
    int ticks_per_step,
    int game_cap_ticks,
    size_t max_players_per_team,
    size_t num_threads,
    const std::vector<int>& worker_cpus
) : pool_(num_threads, worker_cpus) {
    if (games.empty() || num_threads == 0 || game_cap_ticks <= 0) {
        throw std::invalid_argument("EvalVecEnv needs at least one game, one worker thread and a positive game cap.");
    }

    // No no-touch limit: like in game, a game only ends on a goal, or as a draw at the cap.
    const EpisodeLimits limits{0, game_cap_ticks};

    envs_.reserve(games.size());
    size_t agent_offset = 0;
    for (size_t g = 0; g < games.size(); g++) {
        EvalGame& game = games[g];
        if (game.team_size == 0 || game.team_size > max_players_per_team) {
            throw std::invalid_argument("EvalVecEnv: game " + std::to_string(g) + " has a team size of "
                                        + std::to_string(game.team_size) + ", which must be in [1, max_players_per_team].");
        }
        if (game.anchor_slot >= num_anchor_slots) {
            throw std::invalid_argument("EvalVecEnv: game " + std::to_string(g) + " plays anchor slot "
                                        + std::to_string(game.anchor_slot) + " of only " + std::to_string(num_anchor_slots) + ".");
        }
        if (!game.setter) {
            throw std::invalid_argument("EvalVecEnv: game " + std::to_string(g) + " has no state setter.");
        }

        // Blue cars first, like `MatchDistribution::to_team_layout`.
        std::vector<ffi::Team> layout(game.team_size, ffi::Team::Blue);
        layout.insert(layout.end(), game.team_size, ffi::Team::Orange);
        const ffi::Team anchor_team = game.learner_team == ffi::Team::Blue ? ffi::Team::Orange : ffi::Team::Blue;

        std::vector<state::StateSetterDistribution> setters;
        setters.emplace_back(std::move(game.setter), 1.0f);

        envs_.push_back(std::make_unique<RocketSimEnv>(
            layout,
            setters,
            std::vector<reward::RewardEntry>{},
            std::vector<metric::MetricEntry>{},
            ticks_per_step,
            limits,
            max_players_per_team,
            game.seed,
            false,
            anchor_team
        ));
        learner_teams_.push_back(game.learner_team);
        seeds_.push_back(game.seed);
        env_agent_offsets_.push_back(agent_offset);
        env_agent_counts_.push_back(layout.size());
        agent_offset += layout.size();
    }

    total_agents_ = agent_offset;
    agent_rows_.resize(total_agents_);
    env_actions_.resize(total_agents_);
    single_obs_size_ = envs_[0]->get_obs_size();
    action_space_size_ = envs_[0]->get_action_space_size();

    // An env's agents are its blue cars, then its orange cars.
    const auto is_learner = [&](size_t env_idx, size_t agent) {
        const ffi::Team team = agent < games[env_idx].team_size ? ffi::Team::Blue : ffi::Team::Orange;
        return team == learner_teams_[env_idx];
    };

    // Learner rows first, in game order.
    size_t next_row = 0;
    learner_rows_before_.reserve(envs_.size() + 1);
    for (size_t i = 0; i < envs_.size(); i++) {
        learner_rows_before_.push_back(next_row);
        for (size_t a = 0; a < env_agent_counts_[i]; a++) {
            if (is_learner(i, a)) agent_rows_[env_agent_offsets_[i] + a] = static_cast<uint32_t>(next_row++);
        }
    }
    learner_rows_before_.push_back(next_row);
    learner_agents_ = next_row;

    // Then each anchor slot's rows, in game order within the slot.
    anchor_rows_.resize(num_anchor_slots);
    anchor_row_begin_.resize(envs_.size());
    for (size_t slot = 0; slot < num_anchor_slots; slot++) {
        anchor_rows_[slot].begin = next_row;
        for (size_t i = 0; i < envs_.size(); i++) {
            if (games[i].anchor_slot != slot) continue;
            anchor_row_begin_[i] = next_row;
            for (size_t a = 0; a < env_agent_counts_[i]; a++) {
                if (!is_learner(i, a)) agent_rows_[env_agent_offsets_[i] + a] = static_cast<uint32_t>(next_row++);
            }
        }
        anchor_rows_[slot].count = next_row - anchor_rows_[slot].begin;
    }

    outcomes_.assign(envs_.size(), GameOutcome::Unfinished);
    game_ticks_.assign(envs_.size(), 0);

    const auto pinned_opts = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32).pinned_memory(true);
    batched_obs_ = torch::zeros({(int64_t)total_agents_, (int64_t)single_obs_size_}, pinned_opts);
    batched_action_masks_ = torch::zeros({(int64_t)total_agents_, (int64_t)action_space_size_}, pinned_opts);

    reset_task_ = [this](size_t i) {
        envs_[i]->reseed(seeds_[i]);
        const auto& result = envs_[i]->reset();
        write_obs_and_masks(i, result.observations, result.action_masks);
    };
    step_task_ = [this](size_t i) {
        if (outcomes_[i] == GameOutcome::Unfinished) step_env(i);
    };
}

// Defined here, where RocketSimEnv is a complete type.
EvalVecEnv::~EvalVecEnv() = default;

void EvalVecEnv::reset(size_t num_games) {
    num_games = std::min(num_games, envs_.size());
    if (num_games == 0) {
        throw std::invalid_argument("EvalVecEnv::reset needs at least one game to play.");
    }
    std::fill(outcomes_.begin(), outcomes_.end(), GameOutcome::Unfinished);
    std::fill(game_ticks_.begin(), game_ticks_.end(), 0);

    played_games_ = num_games;
    pool_.dispatch(reset_task_, 0, played_games_);
    pool_.wait();
    games_left_ = played_games_;
}

void EvalVecEnv::step(const int* batched_actions) {
    if (all_done()) {
        throw std::logic_error("EvalVecEnv::step called with no unfinished games. Call reset first.");
    }

    // Published to the workers by the pool's dispatch.
    current_actions_ptr_ = batched_actions;
    pool_.dispatch(step_task_, 0, played_games_);
    pool_.wait();
    current_actions_ptr_ = nullptr;

    const auto played_end = outcomes_.begin() + static_cast<std::ptrdiff_t>(played_games_);
    games_left_ = static_cast<size_t>(std::count(outcomes_.begin(), played_end, GameOutcome::Unfinished));
}

void EvalVecEnv::step_env(size_t env_idx) {
    const size_t agent_offset = env_agent_offsets_[env_idx];
    const size_t num_agents = env_agent_counts_[env_idx];
    const uint32_t* rows = agent_rows_.data() + agent_offset;

    // Gather the env's actions from its agents' rows.
    int* env_actions_ptr = env_actions_.data() + agent_offset;
    for (size_t a = 0; a < num_agents; ++a) {
        env_actions_ptr[a] = current_actions_ptr_[rows[a]];
    }
    const StepResult result = envs_[env_idx]->step(env_actions_ptr);
    game_ticks_[env_idx] += result.ticks_elapsed;

    if (result.is_terminated) {
        const bool learner_is_blue = learner_teams_[env_idx] == ffi::Team::Blue;
        outcomes_[env_idx] = result.blue_scored == learner_is_blue ? GameOutcome::Win : GameOutcome::Loss;
    } else if (result.is_truncated) {
        // The only limit is the game cap.
        outcomes_[env_idx] = GameOutcome::Draw;
    } else {
        write_obs_and_masks(env_idx, result.observations, result.action_masks);
    }
}

void EvalVecEnv::write_obs_and_masks(size_t env_idx, const std::vector<float>& obs, const std::vector<float>& masks) {
    const uint32_t* rows = agent_rows_.data() + env_agent_offsets_[env_idx];
    float* batched_obs = batched_obs_.data_ptr<float>();
    float* batched_masks = batched_action_masks_.data_ptr<float>();

    for (size_t a = 0; a < env_agent_counts_[env_idx]; ++a) {
        std::memcpy(batched_obs + rows[a] * single_obs_size_, obs.data() + a * single_obs_size_,
                    single_obs_size_ * sizeof(float));
        std::memcpy(batched_masks + rows[a] * action_space_size_, masks.data() + a * action_space_size_,
                    action_space_size_ * sizeof(float));
    }
}

} // namespace buta_ppo::env
