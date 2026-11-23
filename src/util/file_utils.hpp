// buta-ppo/src/util/file_utils.hpp
#pragma once

#include "platform/file.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>

namespace buta_ppo::util {

// Runs `op`, one or more std::filesystem calls, and if it throws because another program has a file open for the moment
// (platform::is_file_in_use_error: on Windows an antivirus scan of a file just written, search indexing, a sync tool),
// runs it again, for up to about 5 seconds. Any other error, or one that outlasts that, is thrown. So on Linux, where
// files are never in use that way, `op` runs once. `op` must be safe to run again after it failed.
template <typename Op>
void retry_while_in_use(Op&& op) {
    using namespace std::chrono_literals;
    const auto give_up = std::chrono::steady_clock::now() + 5s;
    auto delay = std::chrono::milliseconds(10);
    while (true) {
        try {
            op();
            return;
        } catch (const std::filesystem::filesystem_error& e) {
            if (!platform::is_file_in_use_error(e.code())) throw;
            const auto now = std::chrono::steady_clock::now();
            if (now >= give_up) {
                throw std::runtime_error(std::string(e.what()) + ". Another program kept it in use for 5 seconds: "
                                         "a terminal or Explorer window open in that directory, a sync tool, or an "
                                         "antivirus scan.");
            }
            // The last try comes at the deadline.
            std::this_thread::sleep_for(std::min<std::chrono::steady_clock::duration>(delay, give_up - now));
            delay = std::min(delay * 2, std::chrono::milliseconds(500));
        }
    }
}

// std::filesystem::rename, remove and remove_all, retried while the file is in use (see retry_while_in_use).
inline void rename_with_retry(const std::filesystem::path& from, const std::filesystem::path& to) {
    retry_while_in_use([&] { std::filesystem::rename(from, to); });
}
inline void remove_with_retry(const std::filesystem::path& path) {
    retry_while_in_use([&] { std::filesystem::remove(path); });
}
inline void remove_all_with_retry(const std::filesystem::path& path) {
    retry_while_in_use([&] { std::filesystem::remove_all(path); });
}

// Reads a whole text file with LF line endings, whichever it was saved with: CRLF (as Windows writes text files, or an
// editor saved it) becomes LF. So its contents compare and hash the same on every OS. Empty if it can't be read.
inline std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    size_t kept = 0;
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == '\r' && i + 1 < text.size() && text[i + 1] == '\n') continue;
        text[kept++] = text[i];
    }
    text.resize(kept);
    return text;
}

// Writes a text file via a temp file and rename, so an interrupted write never leaves a partial file.
inline void write_text_file(const std::filesystem::path& path, const std::string& text) {
    const std::filesystem::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        out << text;
        out.close();
        if (!out) throw std::runtime_error("Failed to write " + tmp.string());
    }
    rename_with_retry(tmp, path);
}

} // namespace buta_ppo::util