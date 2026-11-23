// buta-ppo/src/platform/file.hpp
#pragma once

#include <system_error>

namespace buta_ppo::platform {

/**
 * @brief Whether a file operation failed only because another program has the file, or one inside the directory, open
 * for the moment, so trying again shortly may succeed (see `util::retry_while_in_use`).
 *
 * Windows doesn't let a file be renamed or deleted while another program has it open without allowing that, and a file
 * deleted while open keeps its name until it's closed. Antivirus scanners (Defender scans every new file), search
 * indexing and sync tools open files briefly, often right after they're written. Always false on Linux, where an open
 * file can be renamed or deleted.
 *
 * @param error The code of a `std::filesystem::filesystem_error`.
 */
bool is_file_in_use_error(const std::error_code& error);

} // namespace buta_ppo::platform
