// buta-ppo/src/env/reward_manager.cpp
#include "reward_manager.hpp"
#include "env/rocketsim_env.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace buta_ppo::reward {

float TeamRewardFunction::get_reward(const env::AgentMeta& agent, const ffi::ArenaState& current_state, const ffi::ArenaState& previous_state) {
    return get_team_reward(agent.team, current_state, previous_state);
}

std::vector<std::string> validate_reward_entries(const std::vector<RewardEntry>& rewards, size_t largest_team, bool has_empty_team) {
    std::vector<std::string> warnings;
    std::unordered_set<std::string> names;

    for (const RewardEntry& entry : rewards) {
        const std::string prefix = "Reward '" + entry.name + "': ";
        const RewardOptions& options = entry.options;

        // Errors.
        if (!names.insert(entry.name).second) {
            throw std::invalid_argument(prefix + "names must be unique, since telemetry is grouped by name.");
        }
        if (!std::isfinite(entry.weight)) {
            throw std::invalid_argument(prefix + "weight must be finite.");
        }
        // Also rejects NaN, which fails every comparison.
        if (!(options.self_share >= 0.0f && options.self_share <= 1.0f)) {
            throw std::invalid_argument(prefix + "self_share must be in [0, 1].");
        }
        if (!(options.opp_scale >= 0.0f && std::isfinite(options.opp_scale))) {
            throw std::invalid_argument(prefix + "opp_scale must be finite and non-negative.");
        }

        // Warnings.
        if (entry.weight == 0.0f) {
            warnings.push_back(prefix + "weight is 0, so it adds nothing to the reward but is still computed every tick.");
        }
        if (options.self_share != 1.0f) {
            if (entry.function->is_team_reward()) {
                warnings.push_back(prefix + "self_share has no effect on a team reward, since teammates already receive the same value.");
            } else if (largest_team <= 1) {
                warnings.push_back(prefix + "self_share has no effect, since no match layout has more than one player per team.");
            }
        }
        if (options.opp_scale > 0.0f && has_empty_team) {
            warnings.push_back(prefix + "opp_scale has no effect in match layouts with an empty team.");
        }
        if (options.opp_scale > 1.0f) {
            warnings.push_back(prefix + "opp_scale is above 1, so opponents' rewards outweigh the agent's own.");
        }
    }

    return warnings;
}

void RewardManager::set_rewards(const std::vector<RewardEntry>& rewards) {
    rewards_ = rewards;
    for (const RewardEntry& entry : rewards_) {
        telemetry_breakdown_[entry.name] = RewardAccumulation{};
    }
}

void RewardManager::reset(const std::vector<env::AgentMeta>& agents, const ffi::ArenaState& initial_state) {
    // Agents are fixed after the env is built, so this only allocates on the first reset.
    num_agents_ = agents.size();
    agent_teams_.resize(num_agents_);
    team_sizes_ = {};
    for (size_t a = 0; a < num_agents_; a++) {
        agent_teams_[a] = static_cast<size_t>(agents[a].team);
        ++team_sizes_[agent_teams_[a]];
    }
    step_raw_.resize(rewards_.size() * num_agents_);
    step_received_.resize(rewards_.size() * num_agents_);

    previous_state_ = initial_state;
    for (auto& entry : rewards_) {
        entry.function->reset(initial_state);
    }
}

void RewardManager::begin_step() {
    std::fill(step_raw_.begin(), step_raw_.end(), 0.0f);
}

void RewardManager::accumulate_tick(const std::vector<env::AgentMeta>& agents, const ffi::ArenaState& current_state) {
    for (auto& entry : rewards_) {
        entry.function->pre_tick(agents, current_state, previous_state_);
    }

    for (size_t e = 0; e < rewards_.size(); e++) {
        RewardEntry& entry = rewards_[e];
        float* raw = step_raw_.data() + e * num_agents_;

        if (entry.function->is_team_reward()) {
            auto& team_function = static_cast<TeamRewardFunction&>(*entry.function);
            accumulate_team_reward(team_function, entry.weight, raw, current_state);
        } else {
            accumulate_agent_reward(*entry.function, entry.weight, raw, agents, current_state);
        }
    }

    previous_state_ = current_state;
}

void RewardManager::accumulate_team_reward(TeamRewardFunction& function, float weight, float* raw, const ffi::ArenaState& current_state) {
    std::array<float, 2> team_values{};
    for (size_t team = 0; team < team_values.size(); team++) {
        if (team_sizes_[team] > 0) {
            team_values[team] = function.get_team_reward(static_cast<ffi::Team>(team), current_state, previous_state_) * weight;
        }
    }

    for (size_t a = 0; a < num_agents_; a++) {
        raw[a] += team_values[agent_teams_[a]];
    }
}

void RewardManager::accumulate_agent_reward(RewardFunction& function, float weight, float* raw,
                                            const std::vector<env::AgentMeta>& agents, const ffi::ArenaState& current_state) {
    for (size_t a = 0; a < num_agents_; a++) {
        raw[a] += function.get_reward(agents[a], current_state, previous_state_) * weight;
    }
}

void RewardManager::end_step(std::vector<float>& rewards, std::optional<ffi::Team> untracked_team) {
    for (size_t e = 0; e < rewards_.size(); e++) {
        RewardEntry& entry = rewards_[e];
        const float* raw = step_raw_.data() + e * num_agents_;
        float* received = step_received_.data() + e * num_agents_;

        apply_options(entry, raw, received);

        for (size_t a = 0; a < num_agents_; a++) {
            rewards[a] += received[a];

            const bool tracked = static_cast<ffi::Team>(agent_teams_[a]) != untracked_team;
            if (tracked) {
                entry.accumulate.raw += raw[a];
                entry.accumulate.magnitude += std::abs(received[a]);
            }
        }
    }
}

void RewardManager::apply_options(const RewardEntry& entry, const float* raw, float* received) const {
    const RewardOptions& options = entry.options;
    if (options.is_default()) {
        std::copy(raw, raw + num_agents_, received);
        return;
    }

    std::array<float, 2> team_totals{};
    for (size_t a = 0; a < num_agents_; a++) {
        team_totals[agent_teams_[a]] += raw[a];
    }

    for (size_t a = 0; a < num_agents_; a++) {
        const size_t team = agent_teams_[a];
        const size_t opp_team = 1 - team;
        const float team_size = static_cast<float>(team_sizes_[team]);
        const float opp_team_size = static_cast<float>(team_sizes_[opp_team]);
        float value = raw[a];

        // self_share: keep that fraction of the agent's own reward, and receive an even split of what its teammates gave up.
        if (team_sizes_[team] > 1) {
            const float teammates_mean = (team_totals[team] - raw[a]) / (team_size - 1.0f);
            value = options.self_share * raw[a] + (1.0f - options.self_share) * teammates_mean;
        }

        // opp_scale: subtract the opponent team's reward.
        if (team_sizes_[opp_team] > 0) {
            // A per-agent reward subtracts the opponent team's total split across this team, so the team totals cancel.
            // A team reward gives every opponent the same value, so that value is subtracted instead of their total.
            const float opp_reward = entry.function->is_team_reward()
                ? team_totals[opp_team] / opp_team_size
                : team_totals[opp_team] / team_size;
            value -= options.opp_scale * opp_reward;
        }

        received[a] = value;
    }
}

void RewardManager::update_telemetry() {
    for (auto& entry : rewards_) {
        telemetry_breakdown_[entry.name] = entry.accumulate;
        entry.accumulate = RewardAccumulation{};
    }
}

} // namespace buta_ppo::reward
