// buta-ppo/src/util/event_file_writer.hpp
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace buta_ppo::util {

/**
 * @brief Appends scalars to a TensorBoard event file, encoding the few Protobuf messages they need by hand.
 *
 * Opening a file first checks every record and repairs one that TensorBoard would stop reading at, since everything
 * appended after such a record would never show up.
 */
class EventFileWriter {
public:
    /**
     * @brief Repairs the file if it needs it, then opens it for appending, creating it if it doesn't exist.
     *
     * An unfinished last record (a process killed mid-write) is cut off. For anything else that fails its checksum,
     * the bytes from that record on are moved to `<stem>.damaged-<unix time>` next to the file.
     *
     * @param path The event file. TensorBoard only loads files with "tfevents" in their name.
     * @throws std::runtime_error or std::filesystem::filesystem_error if the file can't be read, repaired or opened.
     */
    explicit EventFileWriter(const std::filesystem::path& path);

    /**
     * @brief Appends one scalar, which reaches the file by the next `flush` at the latest.
     *
     * @param tag The chart name, with `/` separating groups.
     * @param step The x-axis position.
     * @param value The value. The event format stores scalars as 32-bit floats.
     */
    void add_scalar(std::string_view tag, int64_t step, float value);

    /**
     * @brief Hands everything added so far to the OS.
     *
     * @return False if this or any earlier write failed, e.g. on a full disk. Later writes then fail too.
     */
    bool flush();

    // What opening the file repaired, as a message for the user. Empty if the file was intact.
    const std::string& repair_report() const { return repair_report_; }

private:
    std::string repair_report_;
    std::ofstream file_;
};

} // namespace buta_ppo::util
