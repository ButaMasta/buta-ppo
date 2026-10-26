// buta-ppo/src/rl/nexto_ladder.hpp
#pragma once

#include "external/nexto/nexto_bot.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace buta_ppo::rl {

/**
 * @brief The evaluation settings a ladder's ratings depend on.
 */
struct NextoLadderSettings {
    std::string model_path; // Nexto's TorchScript file.
    float pos_jitter = 0.0f; // Of the eval kickoffs.
    float yaw_jitter = 0.0f;
    float game_cap_seconds = 0.0f;
};

/**
 * @brief Nexto at a ladder of betas (`NextoOptions::beta`, more random play further down), with every rung's frozen
 * rating per team size: evaluation's fixed reference.
 *
 * Nexto at beta 1, playing its best action, is 1500 Elo in every team size it plays (1v1 to 3v3). Each weaker rung is
 * rated, top down, from Nexto-vs-Nexto games against the rungs above it, already rated. The ratings depend only on Nexto
 * and the start states' distribution (the jittered eval kickoffs and the game cap), not on the learner, so a ladder
 * calibrated once ships with the framework, and one is calibrated at startup only when those settings differ.
 *
 * Ladder files hold the settings they were calibrated with (`fingerprint`), then a `---` line, then one row per rung
 * and team size: `beta team_size rating std_error`, ratings in BT units (× 400 / ln 10 for Elo).
 */
class NextoLadder {
public:
    static constexpr double REFERENCE_ELO = 1500.0;
    static constexpr size_t MAX_TEAM_SIZE = 3; // Nexto plays 1v1 to 3v3.
    // A ladder calibrated for other settings than the shipped one's, in the checkpoint directory.
    static constexpr const char* CALIBRATED_FILE = "nexto_ladder.txt";

    // The rungs' betas, strongest first. The first, beta 1, is the reference.
    [[nodiscard]] static const std::vector<float>& betas();
    // A rung's opponent name in the evaluation ledger, e.g. "nexto_beta_0.1".
    [[nodiscard]] static std::string rung_id(float beta);
    // The settings' lines of a ladder file.
    [[nodiscard]] static std::string fingerprint(const NextoLadderSettings& settings);

    /**
     * @brief How a calibration plays: start states per team size and pairing, each from both sides, and how many
     * rungs above each rung it plays.
     */
    struct Calibration {
        size_t bank_size = 64;
        uint32_t bank_seed = 12345;
        size_t rungs_above = 1;
        size_t num_threads = 1;
        std::vector<int> worker_cpus;
        size_t max_players_per_team = 3; // Sizes the obs the envs build, unused by Nexto.
        int ticks_per_step = 8;
        int virtual_draws = 1;
        // Games built at once, about 2 MiB of memory each. Larger calibrations play in rounds of pairings.
        size_t max_games_at_once = 768;
    };

    /**
     * @brief Plays every rung against the rungs above it in team sizes 1 to 3, and fits their ratings top down.
     * Returns the ladder file's text.
     */
    [[nodiscard]] static std::string calibrate(const NextoLadderSettings& settings, const Calibration& calibration,
                                               const std::shared_ptr<external::NextoBot>& nexto);

    /**
     * @brief Loads the ladder whose settings match: `shipped`, else `calibrated`, else calibrates one (announced on the
     * console) and saves it as `calibrated`.
     */
    void open(const NextoLadderSettings& settings, const std::filesystem::path& shipped,
              const std::filesystem::path& calibrated, const Calibration& calibration,
              const std::shared_ptr<external::NextoBot>& nexto);

    // A rung's frozen rating in a team size (1 to 3), in BT units, and its standard error.
    [[nodiscard]] double rating(size_t rung, size_t team_size) const { return ratings_.at(rung).at(team_size - 1); }
    [[nodiscard]] double std_error(size_t rung, size_t team_size) const { return std_errors_.at(rung).at(team_size - 1); }
    // A hash of the ratings, so a ledger can tell when they change.
    [[nodiscard]] uint64_t hash() const { return hash_; }
    // Where the ladder came from, for the console.
    [[nodiscard]] const std::filesystem::path& source() const { return source_; }

private:
    std::vector<std::vector<double>> ratings_;    // Per rung, per team size - 1.
    std::vector<std::vector<double>> std_errors_;
    uint64_t hash_ = 0;
    std::filesystem::path source_;

    // Parses a ladder file's text if it was calibrated with these settings and has every rung in every team size.
    bool parse(const std::string& text, const NextoLadderSettings& settings);
};

} // namespace buta_ppo::rl
