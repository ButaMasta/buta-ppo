// buta-ppo/src/util/event_file_writer.cpp
#include "event_file_writer.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace fs = std::filesystem;

namespace buta_ppo::util {
namespace {

// Records and fixed-width fields are little-endian, and are written straight from memory.
static_assert(std::endian::native == std::endian::little);

// A record is [u64 length][u32 masked CRC of the length][event][u32 masked CRC of the event].
constexpr uint64_t kHeaderSize = sizeof(uint64_t) + sizeof(uint32_t);
constexpr uint64_t kFooterSize = sizeof(uint32_t);

// Protobuf field keys, (field number << 3) | wire type, of the only fields a scalar uses:
// Event { wall_time = 1, step = 2, summary = 5 }, Summary { value = 1 }, Summary.Value { tag = 1, simple_value = 2 }.
//
// A file deliberately starts without the `file_version` record ("brain.Event:2") other writers put first. Without it,
// TensorBoard's Python loader, the only one on Windows, replaces the newer steps when a run rolls back to an older
// checkpoint. With it, the chart keeps both timelines.
constexpr char kEventWallTime = 0x09;
constexpr char kEventStep = 0x10;
constexpr char kEventSummary = 0x2A;
constexpr char kSummaryValue = 0x0A;
constexpr char kValueTag = 0x0A;
constexpr char kValueSimple = 0x15;

constexpr std::array<uint32_t, 256> kCrc32cTable = [] {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 1) ? (crc >> 1) ^ 0x82F63B78u : crc >> 1;
        }
        table[i] = crc;
    }
    return table;
}();

uint32_t masked_crc32c(const char* data, size_t size) {
    uint32_t crc = ~0u;
    for (size_t i = 0; i < size; ++i) {
        crc = kCrc32cTable[(crc ^ static_cast<unsigned char>(data[i])) & 0xFF] ^ (crc >> 8);
    }
    crc = ~crc;
    return ((crc >> 15) | (crc << 17)) + 0xA282EAD8u;
}

void append_varint(std::string& out, uint64_t value) {
    while (value >= 0x80) {
        out.push_back(static_cast<char>(value | 0x80));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

template <typename T>
void append_raw(std::string& out, T value) {
    out.append(reinterpret_cast<const char*>(&value), sizeof(T));
}

void append_length_delimited(std::string& out, char key, std::string_view bytes) {
    out.push_back(key);
    append_varint(out, bytes.size());
    out.append(bytes);
}

struct ScanResult {
    // The size of the file up to the first bad record.
    uint64_t valid_size;
    // The bad record is an unfinished last one, not a damaged one.
    bool unfinished;
};

ScanResult scan_records(const fs::path& path, uint64_t file_size) {
    std::ifstream in(path, std::ios::binary);
    const auto read = [&](char* destination, uint64_t size) {
        if (!in.read(destination, static_cast<std::streamsize>(size))) {
            throw std::runtime_error("Failed to read TensorBoard log " + path.string());
        }
    };
    if (!in) {
        throw std::runtime_error("Failed to open TensorBoard log " + path.string());
    }

    std::string event;
    uint64_t offset = 0;
    while (offset < file_size) {
        const uint64_t remaining = file_size - offset;
        if (remaining < kHeaderSize) return {offset, true};

        char header[kHeaderSize];
        read(header, kHeaderSize);
        uint64_t length;
        uint32_t length_crc;
        std::memcpy(&length, header, sizeof(length));
        std::memcpy(&length_crc, header + sizeof(length), sizeof(length_crc));
        if (masked_crc32c(header, sizeof(length)) != length_crc) return {offset, false};

        // The length passed its checksum, so a record reaching past the end of the file is one cut short.
        const uint64_t after_header = remaining - kHeaderSize;
        if (after_header < kFooterSize || length > after_header - kFooterSize) return {offset, true};

        event.resize(length);
        read(event.data(), length);
        uint32_t event_crc;
        read(reinterpret_cast<char*>(&event_crc), sizeof(event_crc));
        if (masked_crc32c(event.data(), event.size()) != event_crc) return {offset, false};

        offset += kHeaderSize + length + kFooterSize;
    }
    return {offset, false};
}

void move_tail_aside(const fs::path& path, uint64_t tail_start, const fs::path& destination) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(tail_start));
    std::ofstream out(destination, std::ios::binary | std::ios::trunc);
    out << in.rdbuf();
    out.flush();
    if (!in || !out) {
        throw std::runtime_error("Failed to save the damaged part of TensorBoard log " + path.string() + " to " +
                                 destination.string());
    }
}

// Makes the file end at its last good record, and returns a message for the user if that changed it.
std::string repair_event_file(const fs::path& path) {
    if (!fs::exists(path)) return {};

    const uint64_t file_size = fs::file_size(path);
    const ScanResult scan = scan_records(path, file_size);
    if (scan.valid_size == file_size) return {};

    const std::string removed = std::to_string(file_size - scan.valid_size) + " bytes";
    if (scan.unfinished) {
        fs::resize_file(path, scan.valid_size);
        return "Removed an unfinished record (" + removed + ") from the end of " + path.string() +
               ", left by a run that didn't shut down cleanly.";
    }

    // Named without "tfevents", so TensorBoard doesn't try to load it.
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const fs::path damaged =
        path.parent_path() / (path.stem().string() + ".damaged-" +
                              std::to_string(std::chrono::duration_cast<std::chrono::seconds>(now).count()));
    move_tail_aside(path, scan.valid_size, damaged);
    fs::resize_file(path, scan.valid_size);
    return path.string() + " is damaged at byte " + std::to_string(scan.valid_size) +
           ", which hides everything after it from TensorBoard. Moved those " + removed + " to " + damaged.string() +
           ".";
}

} // namespace

EventFileWriter::EventFileWriter(const fs::path& path) : repair_report_(repair_event_file(path)) {
    file_.open(path, std::ios::binary | std::ios::app);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open TensorBoard log " + path.string());
    }
}

void EventFileWriter::add_scalar(std::string_view tag, int64_t step, float value) {
    std::string summary_value;
    append_length_delimited(summary_value, kValueTag, tag);
    summary_value.push_back(kValueSimple);
    append_raw(summary_value, value);

    std::string summary;
    append_length_delimited(summary, kSummaryValue, summary_value);

    const double wall_time =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::string event;
    event.push_back(kEventWallTime);
    append_raw(event, wall_time);
    // Protobuf leaves out fields holding their default value.
    if (step != 0) {
        event.push_back(kEventStep);
        append_varint(event, static_cast<uint64_t>(step));
    }
    append_length_delimited(event, kEventSummary, summary);

    const uint64_t length = event.size();
    const uint32_t length_crc = masked_crc32c(reinterpret_cast<const char*>(&length), sizeof(length));
    const uint32_t event_crc = masked_crc32c(event.data(), event.size());
    file_.write(reinterpret_cast<const char*>(&length), sizeof(length));
    file_.write(reinterpret_cast<const char*>(&length_crc), sizeof(length_crc));
    file_.write(event.data(), static_cast<std::streamsize>(event.size()));
    file_.write(reinterpret_cast<const char*>(&event_crc), sizeof(event_crc));
}

bool EventFileWriter::flush() {
    file_.flush();
    return file_.good();
}

} // namespace buta_ppo::util
