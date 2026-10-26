// buta-ppo/src/rl/gauntlet.cpp
#include "gauntlet.hpp"
#include "state/setters.hpp"
#include "rating.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <numbers>

namespace fs = std::filesystem;

namespace buta_ppo::rl {

namespace {
constexpr const char* ANCHOR_PREFIX = "snapshot_";
constexpr const char* ANCHOR_EXT = ".pt";

// Parses a `snapshot_<steps>.pt` file name. Anything else, like an interrupted `.pt.tmp` save, is rejected.
bool parse_anchor_name(const fs::path& path, int64_t& steps) {
    if (path.extension() != ANCHOR_EXT) return false;
    const std::string stem = path.stem().string();
    if (!stem.starts_with(ANCHOR_PREFIX)) return false;
    const char* first = stem.data() + std::strlen(ANCHOR_PREFIX);
    const char* last = stem.data() + stem.size();
    const auto [ptr, ec] = std::from_chars(first, last, steps);
    return ec == std::errc{} && ptr == last && first != last;
}

std::string join_sizes(const std::vector<int64_t>& sizes) {
    std::string out;
    for (const int64_t size : sizes) {
        out += " " + std::to_string(size);
    }
    return out;
}
} // namespace

Gauntlet::Gauntlet(const EvalConfig& config, const ModelConfig& model_config, std::vector<size_t> team_sizes,
                   size_t num_threads, const std::vector<int>& worker_cpus, torch::Device device)
    : config_(config), model_config_(model_config), team_sizes_(std::move(team_sizes)), device_(device) {

    if (team_sizes_.empty() || config_.bank_size == 0 || config_.active_anchors == 0 || config_.game_cap_seconds <= 0.0f || 
        config_.promote_win_rate <= 0.0f || config_.promote_win_rate > 1.0f || config_.virtual_draws < 1) {
        throw std::invalid_argument("Evaluation needs a team size, a bank size, active anchors, game cap, and virtual draws above 0, "
                                    "and a promote win rate in (0, 1].");
    }
    // Retiring at or below the promotion score would retire the previous newest anchor as soon as the learner passes it.
    if (config_.retire_win_rate <= config_.promote_win_rate || config_.retire_win_rate > 1.0f) {
        throw std::invalid_argument("Evaluation needs a retire win rate above the promote win rate, and at most 1.");
    }
    slot_anchors_.assign(config_.active_anchors, -1);

    std::vector<env::EvalGame> games;
    for (size_t slot = 0; slot < config_.active_anchors; slot++) {
        for (size_t size_idx = 0; size_idx < team_sizes_.size(); size_idx++) {
            for (size_t state = 0; state < config_.bank_size; state++) {
                for (size_t side = 0; side < 2; side++) {
                    env::EvalGame game;
                    game.team_size = team_sizes_[size_idx];
                    game.anchor_slot = slot;
                    game.learner_team = side == 0 ? ffi::Team::Blue : ffi::Team::Orange;
                    game.setter = state::create_eval_kickoff_setter(config_.bank_seed, static_cast<uint32_t>(state), config_.pos_jitter, config_.yaw_jitter);
                    game.seed = static_cast<uint32_t>((size_idx * config_.bank_size + state) * 2 + side);
                    games.push_back(std::move(game));
                }
            }
        }
    }

    // The cap in physics ticks.
    const int cap_ticks = static_cast<int>(std::lround(config_.game_cap_seconds * 120.0f));
    env_ = std::make_unique<env::EvalVecEnv>(
        std::move(games), config_.active_anchors, static_cast<int>(model_config_.ticks_per_step), cap_ticks,
        static_cast<size_t>(model_config_.max_players_per_team), num_threads, worker_cpus);

    // One inference-only network per anchor slot, like the runner's PFSP opponent slots.
    for (size_t slot = 0; slot < config_.active_anchors; slot++) {
        ActorCritic net(model_config_.network);
        net->to(device_);
        for (auto& param : net->parameters()) {
            param.requires_grad_(false);
        }
        slots_.push_back(net);
    }

    // Buffers for every row: obs and masks are gathered into pinned staging and go up to the GPU, actions come back to
    // a pinned CPU buffer.
    const auto rows = static_cast<int64_t>(env_->get_total_agents());
    const auto obs_size = static_cast<int64_t>(env_->get_single_obs_size());
    const auto action_size = static_cast<int64_t>(env_->get_action_space_size());
    const auto gpu = torch::TensorOptions().device(device_);
    const auto pinned = torch::TensorOptions().pinned_memory(true);
    row_index_.resize(static_cast<size_t>(rows));
    obs_staging_ = torch::empty({rows, obs_size}, pinned.dtype(torch::kFloat32));
    masks_staging_ = torch::empty({rows, action_size}, pinned.dtype(torch::kFloat32));
    obs_gpu_ = torch::empty({rows, obs_size}, gpu.dtype(torch::kFloat32));
    masks_gpu_ = torch::empty({rows, action_size}, gpu.dtype(torch::kFloat32));
    actions_gpu_ = torch::empty({rows}, gpu.dtype(torch::kInt32));
    actions_cpu_ = torch::empty({rows}, pinned.dtype(torch::kInt32));
    step_actions_.resize(static_cast<size_t>(rows));
}

std::string Gauntlet::anchor_id(int64_t steps) {
    return ANCHOR_PREFIX + std::to_string(steps);
}

fs::path Gauntlet::anchor_path(int64_t steps) const {
    return dir_ / (anchor_id(steps) + ANCHOR_EXT);
}

std::string Gauntlet::meta_text() const {
    const ActorCriticConfig& network = model_config_.network;
    return "reference first_snapshot\n"
           "bank_seed " + std::to_string(config_.bank_seed) + "\n" +
           "bank_size " + std::to_string(config_.bank_size) + "\n" +
           "pos_jitter " + std::to_string(config_.pos_jitter) + "\n" +
           "yaw_jitter " + std::to_string(config_.yaw_jitter) + "\n" +
           "game_cap_seconds " + std::to_string(config_.game_cap_seconds) + "\n" +
           "virtual_draws " + std::to_string(config_.virtual_draws) + "\n" +
           "ticks_per_step " + std::to_string(model_config_.ticks_per_step) + "\n" +
           "max_players_per_team " + std::to_string(model_config_.max_players_per_team) + "\n" +
           "obs_size " + std::to_string(network.obs_size) + "\n" +
           "action_size " + std::to_string(network.action_size) + "\n" +
           "shared_layers" + join_sizes(network.shared_layers) + "\n" +
           "actor_layers" + join_sizes(network.actor_layers) + "\n" +
           "critic_layers" + join_sizes(network.critic_layers) + "\n" +
           "use_layer_norm " + std::to_string(network.use_layer_norm ? 1 : 0) + "\n";
}

void Gauntlet::open(const fs::path& dir, int64_t max_steps) {
    dir_ = dir;
    anchors_.clear();
    // Moves a ledger saved with other settings aside, anchor files and all, and drops rows past max_steps.
    ledger_.open(dir_, meta_text(), max_steps);

    for (const auto& entry : fs::directory_iterator(dir_)) {
        int64_t steps = 0;
        if (!entry.is_regular_file() || !parse_anchor_name(entry.path(), steps)) continue;

        if (steps > max_steps) {
            fs::remove(entry.path());
            std::cout << "Deleted Evaluation Anchor " << entry.path().filename().string()
                      << ", which is newer than the resumed checkpoint.\n";
            continue;
        }

        bool rating_exists = false;
        for (const auto& rating : ledger_.ratings()) {
            if (rating.learner_steps == steps) {
                rating_exists = true;
                break;
            }
        }
        if (!rating_exists) {
            fs::remove(entry.path());
            std::cout << "Deleted Evaluation Anchor " << entry.path().filename().string()
                      << ", which has no rating values.\n";
            continue;
        }

        anchors_.push_back(steps);
    }

    std::sort(anchors_.begin(), anchors_.end());

    // The slots' weights may belong to another directory's anchors.
    std::fill(slot_anchors_.begin(), slot_anchors_.end(), -1);
    select_active();
    std::cout << "Loaded " << anchors_.size() << " evaluation anchors from " << dir_.string() << "\n";
}

std::optional<double> Gauntlet::predicted_score(int64_t anchor) const {
    double score_sum = 0.0;
    int rated_sizes = 0;
    for (const size_t size : team_sizes_) {
        const int team_size = static_cast<int>(size);
        const std::optional<EvalLedger::Rating> anchor_rating = ledger_.find_rating(anchor, team_size);
        // The learner's latest rating in this size: the last row with it, since rows are in time order.
        const EvalLedger::Rating* learner_rating = nullptr;
        for (const EvalLedger::Rating& row : ledger_.ratings()) {
            if (row.team_size == team_size) learner_rating = &row;
        }
        if (!anchor_rating || !learner_rating) continue;

        score_sum += 1.0 / (1.0 + std::exp(anchor_rating->rating - learner_rating->rating));
        rated_sizes++;
    }
    if (rated_sizes == 0) return std::nullopt;
    return score_sum / rated_sizes;
}

void Gauntlet::select_active() {
    active_.clear();
    for (auto it = anchors_.rbegin(); it != anchors_.rend() && active_.size() < config_.active_anchors; ++it) {
        const std::optional<double> score = predicted_score(*it);
        if (it == anchors_.rbegin() || !score || *score <= config_.retire_win_rate) {
            active_.push_back(*it);
        }
    }
    std::reverse(active_.begin(), active_.end());

    for (size_t slot = 0; slot < active_.size(); slot++) {
        if (slot_anchors_[slot] == active_[slot]) continue;
        slots_[slot]->load_weights(anchor_path(active_[slot]).string());
        slot_anchors_[slot] = active_[slot];
    }
}

EvalResult Gauntlet::run(int64_t learner_steps, ActorCritic learner) {
    const auto start = std::chrono::steady_clock::now();
    EvalResult result;
    if (anchors_.empty()) {
        // The first anchor is the reference, rated 0 in every team size. Its file is saved before the ledger, so an
        // interrupted save leaves an orphan file that `open` deletes, never a rating without its anchor.
        for (const size_t size : team_sizes_) {
            ledger_.add_rating({learner_steps, static_cast<int>(size), 0.0, 0.0});
            result.sizes.push_back({.team_size = size});
        }
        promote(learner_steps, learner);
        ledger_.save();
        result.promoted = true;
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return result;
    }


    // The anchors to play this time, loaded into the first slots. Only those slots' games are played: a prefix.
    select_active();
    result.anchors_played = active_;
    const size_t real_slots = active_.size();
    const size_t games_per_slot = team_sizes_.size() * config_.bank_size * 2;
    const size_t played = real_slots * games_per_slot;

    // Only the rows of games still running go through the networks. Most games end long before the slowest, so
    // finished games' rows would be most of the work. The running games shrink the same way every time the same games
    // are played, so the results stay reproducible.
    // `segments` holds the compact rows' ranges: the learner's [segments[0], segments[1]), then slot s's
    // [segments[s + 1], segments[s + 2]).
    std::vector<size_t> segments(real_slots + 2, 0);
    size_t gathered_for = SIZE_MAX; // The running game count the rows were gathered for.
    const auto gather_running_rows = [&] {
        const auto& outcomes = env_->outcomes();
        size_t n = 0;
        const auto add = [&](env::RowRange rows) {
            for (size_t k = 0; k < rows.count; k++) row_index_[n++] = rows.begin + k;
        };
        for (size_t game = 0; game < played; game++) {
            if (outcomes[game] == env::GameOutcome::Unfinished) add(env_->game_learner_rows(game));
        }
        segments[1] = n;
        for (size_t slot = 0; slot < real_slots; slot++) {
            for (size_t game = slot * games_per_slot; game < (slot + 1) * games_per_slot; game++) {
                if (outcomes[game] == env::GameOutcome::Unfinished) add(env_->game_anchor_rows(game));
            }
            segments[slot + 2] = n;
        }
    };

    // Each step: gather the running rows' obs and masks, upload them, let each network pick the actions for its own
    // rows, bring the actions back, scatter them to their rows, step the envs. The uploads don't block the CPU; the
    // copy back to `actions_cpu_` does, so the actions are ready, and the staging buffers free, before the next step.
    torch::NoGradGuard no_grad;
    const auto infer = [&](ActorCritic& net, size_t begin, size_t end) {
        const auto first = static_cast<int64_t>(begin);
        const auto count = static_cast<int64_t>(end - begin);
        actions_gpu_.narrow(0, first, count).copy_(
            net->best_action(obs_gpu_.narrow(0, first, count), masks_gpu_.narrow(0, first, count)));
    };
    const size_t obs_size = env_->get_single_obs_size();
    const size_t action_size = env_->get_action_space_size();
    env_->reset(played);
    while (!env_->all_done()) {
        if (env_->games_left() != gathered_for) {
            gather_running_rows();
            gathered_for = env_->games_left();
        }
        const size_t rows = segments[real_slots + 1];
        const float* obs = env_->obs().data_ptr<float>();
        const float* masks = env_->action_masks().data_ptr<float>();
        float* obs_staging = obs_staging_.data_ptr<float>();
        float* masks_staging = masks_staging_.data_ptr<float>();
        for (size_t i = 0; i < rows; i++) {
            std::memcpy(obs_staging + i * obs_size, obs + row_index_[i] * obs_size, obs_size * sizeof(float));
            std::memcpy(masks_staging + i * action_size, masks + row_index_[i] * action_size, action_size * sizeof(float));
        }
        const auto count = static_cast<int64_t>(rows);
        obs_gpu_.narrow(0, 0, count).copy_(obs_staging_.narrow(0, 0, count), true);
        masks_gpu_.narrow(0, 0, count).copy_(masks_staging_.narrow(0, 0, count), true);

        infer(learner, segments[0], segments[1]);
        for (size_t slot = 0; slot < real_slots; slot++) {
            infer(slots_[slot], segments[slot + 1], segments[slot + 2]);
        }
        actions_cpu_.narrow(0, 0, count).copy_(actions_gpu_.narrow(0, 0, count));

        const int* actions = actions_cpu_.data_ptr<int>();
        for (size_t i = 0; i < rows; i++) {
            step_actions_[row_index_[i]] = actions[i];
        }
        env_->step(step_actions_.data());
    }

    // The learner's results per real slot and team size.
    struct Tally {
        int wins = 0;
        int losses = 0;
        int draws = 0;
        int64_t ticks = 0;
    };
    std::vector<std::vector<Tally>> tally(real_slots, std::vector<Tally>(team_sizes_.size()));

    const auto& outcomes = env_->outcomes();
    const auto& game_ticks = env_->game_ticks();
    for (size_t game = 0; game < env_->played_games(); game++) {
        const size_t slot = game / games_per_slot;
        const size_t size_idx = (game % games_per_slot) / (config_.bank_size * 2);

        Tally& t = tally[slot][size_idx];
        if (outcomes[game] == env::GameOutcome::Win) {
            t.wins++;
        } else if (outcomes[game] == env::GameOutcome::Loss) {
            t.losses++;
        } else {
            t.draws++;
        }
        t.ticks += game_ticks[game];

    }

    // Every result goes into the ledger. Each team size is rated against the anchors that have a frozen rating in it.
    constexpr double ELO_PER_RATING = 400.0 / std::numbers::ln10;
    int newest_wins = 0;   // Against the newest anchor, over all team sizes, for promotion.
    int newest_losses = 0;
    int newest_draws = 0;

    for (size_t size_idx = 0; size_idx < team_sizes_.size(); size_idx++) {
        const int size = static_cast<int>(team_sizes_[size_idx]);
        EvalSizeResult& size_result = result.sizes.emplace_back();
        size_result.team_size = team_sizes_[size_idx];

        std::vector<AnchorResult> rated;
        int64_t ticks = 0;
        for (size_t slot = 0; slot < real_slots; slot++) {
            const int64_t anchor_steps = active_[slot];
            const Tally& t = tally[slot][size_idx];
            ledger_.add_result({learner_steps, size, anchor_id(anchor_steps), t.wins, t.losses, t.draws});
            size_result.wins += t.wins;
            size_result.losses += t.losses;
            size_result.draws += t.draws;
            ticks += t.ticks;
            if (const auto anchor_rating = ledger_.find_rating(anchor_steps, size)) {
                rated.push_back({anchor_rating->rating, t.wins, t.losses, t.draws});
            }
            if (slot == real_slots - 1) {
                newest_wins += t.wins;
                newest_losses += t.losses;
                newest_draws += t.draws;
            }
        }
        const int games = size_result.wins + size_result.losses + size_result.draws;
        size_result.mean_game_seconds = static_cast<double>(ticks) / 120.0 / games;

        RatingFit fit{0.0, 0.0}; // The reference's rating, used when no anchor is rated in this size.
        if (!rated.empty()) {
            double start_rating = 0.0;
            for (const EvalLedger::Rating& row : ledger_.ratings()) {
                if (size == row.team_size) {
                    start_rating = row.rating;
                }
            }
            fit = fit_learner_rating(rated, start_rating, config_.virtual_draws);
        }
        ledger_.add_rating({learner_steps, size, fit.rating, fit.std_error});
        size_result.elo = fit.rating * ELO_PER_RATING;
        size_result.elo_std_error = fit.std_error * ELO_PER_RATING;

    }

    if (((newest_wins + newest_draws * 0.5) / (newest_wins + newest_losses + newest_draws)) >= config_.promote_win_rate) {
        promote(learner_steps, learner);
        result.promoted = true;
    }

    ledger_.save();
    result.played = true;
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

void Gauntlet::promote(int64_t steps, const ActorCritic& learner) {
    // Saved via a temp file and rename, so an interrupted save is never loaded.
    const fs::path path = anchor_path(steps);
    const fs::path tmp = path.string() + ".tmp";
    save_model(learner, model_config_, tmp.string());
    fs::rename(tmp, path);
    anchors_.push_back(steps);
}


} // namespace buta_ppo::rl