// buta-ppo/src/rl/eval_ledger.cpp
#include "eval_ledger.hpp"
#include "util/file_utils.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <optional>

namespace fs = std::filesystem;

namespace buta_ppo::rl {

namespace {
constexpr const char* META_FILE = "meta.txt";
constexpr const char* RESULTS_FILE = "results.txt";
constexpr const char* RATINGS_FILE = "ratings.txt";
} // namespace

void EvalLedger::open(const fs::path& dir, const std::string& meta, int64_t max_steps) {
    dir_ = dir;
    results_.clear();
    ratings_.clear();

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
            std::cout << "Evaluation Ledger at " << dir_.string() << " was saved with different evaluation settings. "
                        << "Moved it to " << stale.string() << " and started a fresh ledger.\n";
        }
    }


    fs::create_directories(dir_);
    util::write_text_file(dir_ / META_FILE, meta);

    if (std::ifstream in(dir_ / RESULTS_FILE); in) {
        Result row;
        while (in >> row.learner_steps >> row.team_size >> row.opponent >> row.wins >> row.losses >> row.draws) {
            // TODO 2: keep the row unless it's newer than max_steps.
            if (row.learner_steps > max_steps) continue;
            results_.push_back(row);
        }
    }

    if (std::ifstream in(dir_ / RATINGS_FILE); in) {
        Rating row;
        while (in >> row.learner_steps >> row.team_size >> row.rating >> row.std_error) {
            // TODO 2: keep the row unless it's newer than max_steps.
            if (row.learner_steps > max_steps) continue;
            ratings_.push_back(row);
        }
    }

    // Rewrite both files, so the dropped rows are gone from disk too.
    save();
    std::cout << "Loaded " << ratings_.size() << " evaluation ratings from " << dir_.string() << "\n";
}


void EvalLedger::add_result(const Result& result) {
    results_.push_back(result);
}

void EvalLedger::add_rating(const Rating& rating) {
    ratings_.push_back(rating);
}

std::optional<EvalLedger::Rating> EvalLedger::find_rating(int64_t learner_steps, int team_size) const {
    for (const Rating& rating : ratings_) {
        if (rating.learner_steps == learner_steps && rating.team_size == team_size) {
            return rating;
        }
    }
    return std::nullopt;
}

void EvalLedger::save() const {
    if (dir_.empty()) return;

    std::ostringstream out_results;
    out_results << std::setprecision(17);
    for (const Result& result : results_) {
        out_results << result.learner_steps << " " << result.team_size << " " << result.opponent 
        << " " << result.wins << " " << result.losses << " " << result.draws << "\n";
    }
    util::write_text_file(dir_ / RESULTS_FILE, out_results.str());

    std::ostringstream out_ratings;
    out_ratings << std::setprecision(17);
    for (const Rating& rating : ratings_) {
        out_ratings << rating.learner_steps << " " << rating.team_size << " " << rating.rating << " " << rating.std_error << "\n";
    }
    util::write_text_file(dir_ / RATINGS_FILE, out_ratings.str());
}

} // namespace buta_ppo::rl
