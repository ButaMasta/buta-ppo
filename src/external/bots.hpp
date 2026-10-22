// buta-ppo/src/external/bots.hpp
#pragma once

#include "external_bot.hpp"
#include "nexto/nexto_bot.hpp"

#include <memory>
#include <string>

namespace buta_ppo::external {

/**
 * @brief Nexto, from the botpack's TorchScript `nexto-model.pt`. See `nexto/nexto_bot.hpp`.
 */
inline std::shared_ptr<ExternalBot> create_nexto_bot(const std::string& model_path, const NextoOptions& options = {}) {
    return std::make_shared<NextoBot>(model_path, options);
}

} // namespace buta_ppo::external
