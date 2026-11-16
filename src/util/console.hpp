// buta-ppo/src/util/console.hpp
#pragma once

#include <locale>
#include <ostream>
#include <string>

namespace buta_ppo::util {

// Groups the digits of numbers written to a stream in threes with commas (1,234,567.5), as en_US does, but without
// needing a named locale: those names differ between OSes ("en_US.utf8" on Linux, "en-US" on Windows), and a system may
// not have the locale installed. Everything but the grouping stays as in the stream's current locale.
inline void use_thousands_separators(std::ostream& stream) {
    struct ThousandsSeparators : std::numpunct<char> {
        char do_thousands_sep() const override { return ','; }
        std::string do_grouping() const override { return "\3"; }
    };
    // The locale owns the facet and deletes it with the last locale that uses it.
    stream.imbue(std::locale(stream.getloc(), new ThousandsSeparators));
}

} // namespace buta_ppo::util
