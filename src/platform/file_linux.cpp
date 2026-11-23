// buta-ppo/src/platform/file_linux.cpp
#include "platform/file.hpp"

namespace buta_ppo::platform {

// An open file doesn't stop a rename or delete on Linux, so no error here goes away by waiting.
bool is_file_in_use_error(const std::error_code&) {
    return false;
}

} // namespace buta_ppo::platform
