// buta-ppo/src/rl/rating.cpp
#include "rating.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace buta_ppo::rl {

namespace {

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

} // namespace

RatingFit fit_learner_rating(const std::vector<AnchorResult>& anchors, double start_rating, int virtual_draws) {
    // W: the learner's actual wins, counting each draw (real or virtual) as half a win.
    double actual_wins = 0.0;
    double total_games = 0.0;
    for (const AnchorResult& a : anchors) {
        actual_wins += a.wins + 0.5 * a.draws + 0.5 * virtual_draws;
        total_games += a.wins + a.draws + a.losses + virtual_draws;
    }
    if (actual_wins <= 0 || actual_wins >= total_games) {
        throw std::invalid_argument("fit_learner_rating: no finite solution for the given actual wins.");
    }

    double r = start_rating;
    for (int step = 0; step < 100; step++) {
        double expected_wins = 0.0;
        double info = 0.0;
        for (const AnchorResult& a : anchors) {
            double n = a.wins + a.draws + a.losses + virtual_draws;
            double p = sigmoid(r - a.rating);
            double n_p = n * p;
            expected_wins += n_p;
            info += n_p * ( 1 - p );
        }

        double move = (-1 * (expected_wins - actual_wins)) / info;
        move = std::clamp(move, -1.0, 1.0);
        r += move;
        if (std::abs(move) < 1e-6) {
            return {r, 1.0 / std::sqrt(info)};
        }
    }
    throw std::runtime_error("fit_learner_rating: no convergence after 100 steps");
}

} // namespace buta_ppo::rl
