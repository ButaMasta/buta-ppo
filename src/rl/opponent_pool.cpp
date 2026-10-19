// buta-ppo/src/rl/opponent_pool.cpp
#include "opponent_pool.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace fs = std::filesystem;

namespace buta_ppo::rl {

namespace {
constexpr const char* STATE_FILE = "state.txt";
constexpr const char* META_FILE = "meta.txt";
constexpr const char* SNAPSHOT_EXT = ".pt";

fs::path snapshot_path(const fs::path& dir, int64_t steps) {
    return dir / (std::to_string(steps) + SNAPSHOT_EXT);
}

// Parses a `<steps>.pt` snapshot file name. Anything else, like an interrupted `<steps>.pt.tmp` save, is rejected.
bool parse_snapshot_name(const fs::path& path, int64_t& steps) {
    if (path.extension() != SNAPSHOT_EXT) return false;
    const std::string stem = path.stem().string();
    const char* first = stem.data();
    const char* last = stem.data() + stem.size();
    const auto [ptr, ec] = std::from_chars(first, last, steps);
    return ec == std::errc{} && ptr == last && first != last;
}

// Writes a text file via a temp file and rename, so an interrupted write never leaves a partial file.
void write_text_file(const fs::path& path, const std::string& text) {
    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        out << text;
    }
    fs::rename(tmp, path);
}

std::string join_sizes(const std::vector<int64_t>& sizes) {
    std::string out;
    for (const int64_t size : sizes) {
        out += " " + std::to_string(size);
    }
    return out;
}
} // namespace

OpponentPool::OpponentPool(const PFSPConfig& config, const ActorCriticConfig& ac_config, uint32_t seed)
    : config_(config), ac_config_(ac_config), rng_(seed) {
    if (config_.pool_size == 0 || config_.snapshot_interval <= 0 || config_.win_rate_window < 1.0f ||
        config_.min_weight <= 0.0f) {
        throw std::invalid_argument("PFSP needs a pool size of at least 1, a positive snapshot interval, "
                                    "a win rate window of at least 1 game, and a positive min weight.");
    }
}

std::string OpponentPool::meta_text() const {
    return "obs_size " + std::to_string(ac_config_.obs_size) + "\n" +
           "action_size " + std::to_string(ac_config_.action_size) + "\n" +
           "shared_layers" + join_sizes(ac_config_.shared_layers) + "\n" +
           "actor_layers" + join_sizes(ac_config_.actor_layers) + "\n" +
           "critic_layers" + join_sizes(ac_config_.critic_layers) + "\n" +
           "use_layer_norm " + std::to_string(ac_config_.use_layer_norm ? 1 : 0) + "\n";
}

void OpponentPool::open(const fs::path& dir, int64_t max_steps) {
    dir_ = dir;
    snapshots_.clear();
    const std::string meta = meta_text();

    // A pool of a different network shape can't be loaded into the opponent slots. Keep it aside, untouched.
    if (fs::exists(dir_)) {
        std::string saved_meta;
        if (std::ifstream in(dir_ / META_FILE); in) {
            saved_meta.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        if (saved_meta != meta) {
            const auto now = std::chrono::system_clock::now().time_since_epoch();
            const fs::path stale = dir_.string() + ".stale-" +
                std::to_string(std::chrono::duration_cast<std::chrono::seconds>(now).count());
            fs::rename(dir_, stale);
            std::cout << "PFSP pool at " << dir_.string() << " was saved with a different network shape. "
                      << "Moved it to " << stale.string() << " and started a fresh pool.\n";
        }
    }
    fs::create_directories(dir_);
    write_text_file(dir_ / META_FILE, meta);

    // Win records by snapshot steps.
    std::unordered_map<int64_t, std::pair<double, double>> records;
    if (std::ifstream in(dir_ / STATE_FILE); in) {
        int64_t steps = 0;
        double wins = 0.0;
        double games = 0.0;
        while (in >> steps >> wins >> games) {
            records[steps] = {wins, games};
        }
    }

    for (const auto& entry : fs::directory_iterator(dir_)) {
        int64_t steps = 0;
        if (!entry.is_regular_file() || !parse_snapshot_name(entry.path(), steps)) continue;

        // A snapshot from past the resumed checkpoint belongs to a run that was rolled back.
        if (steps > max_steps) {
            fs::remove(entry.path());
            std::cout << "Deleted PFSP snapshot " << entry.path().filename().string()
                      << ", which is newer than the resumed checkpoint.\n";
            continue;
        }

        ActorCritic staging(ac_config_);
        try {
            staging->load_weights(entry.path().string());
        } catch (const std::runtime_error& e) {
            std::cout << "Skipping unusable PFSP snapshot: " << e.what() << "\n";
            continue;
        }

        Snapshot snapshot = make_snapshot(steps, staging);
        if (const auto it = records.find(steps); it != records.end()) {
            snapshot.wins = it->second.first;
            snapshot.games = it->second.second;
        }
        snapshots_.push_back(std::move(snapshot));
    }

    std::sort(snapshots_.begin(), snapshots_.end(),
              [](const Snapshot& a, const Snapshot& b) { return a.steps < b.steps; });

    // The pool size may have been lowered since the pool was saved.
    while (snapshots_.size() > config_.pool_size) {
        evict();
    }
    save_state();

    std::cout << "Loaded " << snapshots_.size() << " PFSP snapshots from " << dir_.string() << "\n";
}

void OpponentPool::save_state() const {
    if (dir_.empty()) return;

    std::ostringstream out;
    out << std::setprecision(17);
    for (const Snapshot& snapshot : snapshots_) {
        out << snapshot.steps << " " << snapshot.wins << " " << snapshot.games << "\n";
    }
    write_text_file(dir_ / STATE_FILE, out.str());
}

OpponentPool::Snapshot OpponentPool::make_snapshot(int64_t steps, const ActorCritic& net) {
    Snapshot snapshot;
    snapshot.steps = steps;

    const auto pinned_opts = torch::TensorOptions().device(torch::kCPU).pinned_memory(true);
    torch::NoGradGuard no_grad;
    for (const ParamInfo& info : net->parameter_layout()) {
        if (info.section == ParamSection::Critic) {
            snapshot.params.emplace_back();
            continue;
        }
        snapshot.params.push_back(torch::empty(info.tensor.sizes(), pinned_opts.dtype(info.tensor.dtype())).copy_(info.tensor));
    }
    return snapshot;
}

void OpponentPool::add(int64_t steps, const ActorCritic& learner) {
    snapshots_.push_back(make_snapshot(steps, learner));

    if (!dir_.empty()) {
        // Save the full model, via a temp file and rename so an interrupted save is never loaded.
        const fs::path path = snapshot_path(dir_, steps);
        const fs::path tmp = path.string() + ".tmp";
        torch::save(learner, tmp.string());
        fs::rename(tmp, path);
    }

    if (snapshots_.size() > config_.pool_size) {
        evict();
    }
    save_state();
}

std::vector<size_t> OpponentPool::sample(size_t count) {
    if (snapshots_.empty()) {
        throw std::logic_error("OpponentPool::sample called on an empty pool.");
    }

    std::vector<double> weights;
    weights.reserve(snapshots_.size());
    for (const Snapshot& snapshot : snapshots_) {
        weights.push_back(weight(snapshot));
    }

    // Without replacement while there are enough snapshots, so the slots play different versions.
    const bool distinct = snapshots_.size() >= count;
    std::vector<size_t> picks;
    picks.reserve(count);
    for (size_t i = 0; i < count; i++) {
        std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
        const size_t pick = dist(rng_);
        picks.push_back(pick);
        if (distinct) {
            weights[pick] = 0.0;
        }
    }
    return picks;
}

void OpponentPool::record(size_t idx, uint32_t wins, uint32_t losses) {
    Snapshot& snapshot = snapshots_.at(idx);

    // Each game decays the earlier ones by (1 - 1 / window), so the estimate covers about the last `window` games.
    // The update's games are treated as equally old.
    const uint32_t games = wins + losses;
    const double decay = std::pow(1.0 - 1.0 / static_cast<double>(config_.win_rate_window), static_cast<double>(games));
    snapshot.wins = snapshot.wins * decay + wins;
    snapshot.games = snapshot.games * decay + games;
}

double OpponentPool::weight(const Snapshot& snapshot) const {
    const double p = snapshot.win_rate();
    const double w = (config_.weighting == PFSPWeighting::Hard)
        ? std::pow(1.0 - p, static_cast<double>(config_.hard_exponent))
        : 4.0 * p * (1.0 - p);
    return std::max(w, static_cast<double>(config_.min_weight));
}

void OpponentPool::evict() {
    // Never evict the newest snapshot (the last one).
    size_t victim = 0;
    double best_win_rate = -1.0;
    for (size_t i = 0; i + 1 < snapshots_.size(); i++) {
        const Snapshot& snapshot = snapshots_[i];
        if (snapshot.games >= EVICT_MIN_GAMES && snapshot.win_rate() > best_win_rate) {
            best_win_rate = snapshot.win_rate();
            victim = i;
        }
    }
    // With no snapshot trusted enough, `victim` stays at the oldest.
    if (!dir_.empty()) {
        fs::remove(snapshot_path(dir_, snapshots_[victim].steps));
    }
    snapshots_.erase(snapshots_.begin() + static_cast<std::ptrdiff_t>(victim));
}

} // namespace buta_ppo::rl
