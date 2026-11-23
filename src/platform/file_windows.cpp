// buta-ppo/src/platform/file_windows.cpp
#include "platform/file.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace buta_ppo::platform {

bool is_file_in_use_error(const std::error_code& error) {
    // MSVC's std::filesystem reports the Win32 error code, in the system category.
    if (error.category() != std::system_category()) return false;
    switch (error.value()) {
    case ERROR_SHARING_VIOLATION: // Open elsewhere without allowing this.
    case ERROR_LOCK_VIOLATION:    // Part of it is locked.
    case ERROR_ACCESS_DENIED:     // What Windows usually reports for a file that's open, or deleted but still open.
    case ERROR_DELETE_PENDING:    // Deleted, but still open elsewhere.
    case ERROR_DIR_NOT_EMPTY:     // A file inside is deleted but still open elsewhere, so it's still there.
        return true;
    default:
        return false;
    }
}

} // namespace buta_ppo::platform
