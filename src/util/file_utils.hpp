// buta-ppo/src/util/file_utils.hpp
#pragma once

#include <fstream>
#include <filesystem>
#include <string>

namespace buta_ppo::util {

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