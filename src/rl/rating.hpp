// buta-ppo/src/rl/rating.hpp
#pragma once

#include <vector>

namespace buta_ppo::rl {

// The learner's results against one anchor, whose rating is frozen.
struct AnchorResult {
    double rating = 0.0;
    int wins = 0;
    int losses = 0;
    int draws = 0;
};

struct RatingFit {
    double rating = 0.0;
    double std_error = 0.0;
};

// Fits the learner's rating against frozen anchors. Each anchor also gets `virtual_draws` draws, which keeps the
// rating finite when the learner wins or loses every game.
RatingFit fit_learner_rating(const std::vector<AnchorResult>& anchors, double start_rating, int virtual_draws = 1);

} // namespace buta_ppo::rl
