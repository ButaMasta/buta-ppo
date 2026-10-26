// buta-ppo/src/rl/nexto_ladder.cpp
#include "nexto_ladder.hpp"
#include "rating.hpp"
#include "env/eval_vec_env.hpp"
#include "state/setters.hpp"
#include "util/file_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace buta_ppo::rl {

namespace {
constexpr double ELO_PER_RATING = 400.0 / std::numbers::ln10;
// Bumped whenever the eval kickoff setter changes what it sets up, which changes every rating.
constexpr const char* SETTER_VERSION = "eval_kickoff_v1";

uint64_t fnv1a(const std::string& bytes) {
    uint64_t h = 1469598103934665603ull;
    for (const unsigned char c : bytes) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string hex(uint64_t v) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << v;
    return out.str();
}
} // namespace

const std::vector<float>& NextoLadder::betas() {
    static const std::vector<float> rungs = {1.0f, 0.75f, 0.5f, 0.375f, 0.25f, 0.2f, 0.15f, 0.125f, 0.1f, 0.075f, 0.05f, 0.025f, 0.0f};
    return rungs;
}

std::string NextoLadder::rung_id(float beta) {
    char text[32];
    std::snprintf(text, sizeof text, "%g", static_cast<double>(beta));
    std::string id = text;
    if (id.find('.') == std::string::npos) id += ".0";
    return "nexto_beta_" + id;
}

std::string NextoLadder::fingerprint(const NextoLadderSettings& settings) {
    const std::string model = read_file(settings.model_path);
    if (model.empty()) {
        throw std::invalid_argument("Nexto model not found at '" + settings.model_path + "'.");
    }
    std::ostringstream out;
    out << "model_fnv1a " << hex(fnv1a(model)) << "\n"
        << "setter " << SETTER_VERSION << "\n"
        << "pos_jitter " << std::to_string(settings.pos_jitter) << "\n"
        << "yaw_jitter " << std::to_string(settings.yaw_jitter) << "\n"
        << "game_cap_seconds " << std::to_string(settings.game_cap_seconds) << "\n"
        << "betas";
    for (const float beta : betas()) out << " " << beta;
    out << "\n";
    return out.str();
}

std::string NextoLadder::calibrate(const NextoLadderSettings& settings, const Calibration& calibration,
                                   const std::shared_ptr<external::NextoBot>& nexto) {
    const std::vector<float>& rung_betas = betas();
    const size_t num_rungs = rung_betas.size();
    std::vector<std::shared_ptr<external::NextoBot>> rungs;
    for (const float beta : rung_betas) {
        external::NextoOptions options = nexto->options();
        options.beta = beta;
        rungs.push_back(nexto->with_options(options));
    }

    // Each rung (on the learner's side) against each of the rungs just above it.
    struct Pairing {
        size_t rung;
        size_t above;
    };
    std::vector<Pairing> pairings;
    for (size_t rung = 1; rung < num_rungs; rung++) {
        for (size_t k = 1; k <= calibration.rungs_above && k <= rung; k++) pairings.push_back({rung, rung - k});
    }

    // Results per pairing and team size.
    struct Tally {
        int wins = 0;
        int losses = 0;
        int draws = 0;
    };
    std::vector<std::vector<Tally>> tally(pairings.size(), std::vector<Tally>(MAX_TEAM_SIZE));
    const size_t games_per_pairing = MAX_TEAM_SIZE * calibration.bank_size * 2;
    const size_t pairings_at_once = std::max<size_t>(1, calibration.max_games_at_once / games_per_pairing);
    const int cap_ticks = static_cast<int>(std::lround(settings.game_cap_seconds * 120.0f));

    // One set of envs for `pairings_at_once` pairings, reused for every round of pairings with their bots swapped in,
    // which bounds memory (about 2 MiB per env) whatever the calibration's size.
    // Laid out like the gauntlet's: pairing, team size, bank state, side.
    std::vector<env::EvalGame> games;
    const size_t slots = std::min(pairings_at_once, pairings.size());
    for (size_t p = 0; p < slots; p++) {
        for (size_t size = 1; size <= MAX_TEAM_SIZE; size++) {
            for (size_t state = 0; state < calibration.bank_size; state++) {
                for (size_t side = 0; side < 2; side++) {
                    env::EvalGame game;
                    game.team_size = size;
                    game.learner_team = side == 0 ? ffi::Team::Blue : ffi::Team::Orange;
                    game.learner_bot = rungs[1];
                    game.opponent_bot = rungs[0];
                    game.setter = state::create_eval_kickoff_setter(calibration.bank_seed, static_cast<uint32_t>(state),
                                                                    settings.pos_jitter, settings.yaw_jitter);
                    game.seed = static_cast<uint32_t>(((size - 1) * calibration.bank_size + state) * 2 + side);
                    games.push_back(std::move(game));
                }
            }
        }
    }
    env::EvalVecEnv env(std::move(games), 0, calibration.ticks_per_step, cap_ticks,
                        std::max(calibration.max_players_per_team, MAX_TEAM_SIZE), calibration.num_threads,
                        calibration.worker_cpus);

    const auto start = std::chrono::steady_clock::now();
    for (size_t first = 0; first < pairings.size(); first += slots) {
        const size_t last = std::min(pairings.size(), first + slots);
        for (size_t p = first; p < last; p++) {
            for (size_t g = 0; g < games_per_pairing; g++) {
                const size_t game = (p - first) * games_per_pairing + g;
                env.set_external_bot(game, true, rungs[pairings[p].rung]);
                env.set_external_bot(game, false, rungs[pairings[p].above]);
            }
        }
        const size_t num_games = (last - first) * games_per_pairing;
        env.reset(0, num_games);
        // Nexto plays every car: no rows, no actions.
        const int no_actions = 0;
        while (!env.all_done()) env.step(&no_actions);

        for (size_t g = 0; g < num_games; g++) {
            const size_t p = first + g / games_per_pairing;
            const size_t size_idx = (g % games_per_pairing) / (calibration.bank_size * 2);
            Tally& t = tally[p][size_idx];
            switch (env.outcomes()[g]) {
                case env::GameOutcome::Win: t.wins++; break;
                case env::GameOutcome::Loss: t.losses++; break;
                default: t.draws++; break;
            }
        }
        std::cout << "Nexto Ladder Calibration: played " << last << " of " << pairings.size() << " pairings ("
                  << std::fixed << std::setprecision(0)
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s)\n"
                  << std::defaultfloat;
    }

    // Top down: each rung's rating against the rungs above it, already rated.
    std::ostringstream rows;
    rows << std::setprecision(10);
    for (size_t size = 1; size <= MAX_TEAM_SIZE; size++) {
        std::vector<double> rating(num_rungs, REFERENCE_ELO / ELO_PER_RATING);
        std::vector<double> std_error(num_rungs, 0.0);
        for (size_t rung = 1; rung < num_rungs; rung++) {
            std::vector<AnchorResult> results;
            std::vector<double> anchor_se, information;
            for (size_t p = 0; p < pairings.size(); p++) {
                if (pairings[p].rung != rung) continue;
                const Tally& t = tally[p][size - 1];
                results.push_back({rating[pairings[p].above], t.wins, t.losses, t.draws});
                anchor_se.push_back(std_error[pairings[p].above]);
            }
            const RatingFit fit = fit_learner_rating(results, rating[rung - 1], calibration.virtual_draws);
            rating[rung] = fit.rating;
            // The fit treats the rungs above as exact. Their own errors carry down, weighted by how much each pairing
            // told, and taken as fully correlated (they come from the same chain), which overstates rather than hides.
            double weight_sum = 0.0, carried = 0.0;
            for (size_t k = 0; k < results.size(); k++) {
                const AnchorResult& r = results[k];
                const double games = r.wins + r.losses + r.draws + calibration.virtual_draws;
                const double p = 1.0 / (1.0 + std::exp(r.rating - fit.rating));
                const double info = games * p * (1.0 - p);
                weight_sum += info;
                carried += info * anchor_se[k];
            }
            carried = weight_sum > 0.0 ? carried / weight_sum : 0.0;
            std_error[rung] = std::sqrt(fit.std_error * fit.std_error + carried * carried);
        }
        for (size_t rung = 0; rung < num_rungs; rung++) {
            rows << betas()[rung] << " " << size << " " << rating[rung] << " " << std_error[rung] << "\n";
        }
    }

    std::ostringstream text;
    text << "# Nexto ladder: Nexto's rating at each beta, per team size, with Nexto at beta 1 fixed at "
         << REFERENCE_ELO << " Elo.\n"
         << "# Ratings in BT units (x 400 / ln 10 for Elo). Calibrated with " << calibration.bank_size
         << " kickoffs per team size and pairing, each from both sides, against the " << calibration.rungs_above
         << " rung(s) above.\n"
         << fingerprint(settings) << "---\n" << rows.str();
    return text.str();
}

bool NextoLadder::parse(const std::string& text, const NextoLadderSettings& settings) {
    // Comment lines first, then the settings, then the rows.
    std::istringstream in(text);
    std::string line, header;
    while (std::getline(in, line) && line != "---") {
        if (line.empty() || line[0] == '#') continue;
        header += line + "\n";
    }
    if (header != fingerprint(settings)) return false;

    const size_t num_rungs = betas().size();
    std::vector<std::vector<double>> ratings(num_rungs, std::vector<double>(MAX_TEAM_SIZE, NAN));
    std::vector<std::vector<double>> std_errors = ratings;
    std::string rows;
    float beta = 0.0f;
    size_t size = 0;
    double rating = 0.0, std_error = 0.0;
    while (in >> beta >> size >> rating >> std_error) {
        const auto it = std::find(betas().begin(), betas().end(), beta);
        if (it == betas().end() || size < 1 || size > MAX_TEAM_SIZE) return false;
        const auto rung = static_cast<size_t>(it - betas().begin());
        ratings[rung][size - 1] = rating;
        std_errors[rung][size - 1] = std_error;
    }
    for (const auto& per_size : ratings) {
        for (const double r : per_size) {
            if (std::isnan(r)) return false;
        }
    }
    ratings_ = std::move(ratings);
    std_errors_ = std::move(std_errors);
    hash_ = fnv1a(text.substr(text.find("---")));
    return true;
}

void NextoLadder::open(const NextoLadderSettings& settings, const fs::path& shipped, const fs::path& calibrated,
                       const Calibration& calibration, const std::shared_ptr<external::NextoBot>& nexto) {
    for (const fs::path& path : {shipped, calibrated}) {
        if (fs::exists(path) && parse(read_file(path), settings)) {
            source_ = path;
            return;
        }
    }

    std::cout << "Nexto Ladder: no calibration matches these evaluation settings (Nexto model, kickoff jitter, game cap), "
              << "so it's calibrated now, once: about a minute. It's saved as " << calibrated.string()
              << " and reused while the settings stay the same.\n";
    const std::string text = calibrate(settings, calibration, nexto);
    if (!parse(text, settings)) {
        throw std::logic_error("Nexto Ladder: a fresh calibration didn't parse.");
    }
    if (!calibrated.parent_path().empty()) fs::create_directories(calibrated.parent_path());
    util::write_text_file(calibrated, text);
    source_ = calibrated;
}

} // namespace buta_ppo::rl
