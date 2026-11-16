// buta-ppo/src/util/file_utils.hpp
#pragma once

#include <fstream>
#include <filesystem>
#include <iterator>
#include <string>

namespace buta_ppo::util {

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
    }
    std::filesystem::rename(tmp, path);
}

} // namespace buta_ppo::util