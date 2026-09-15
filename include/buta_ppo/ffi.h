// buta-ppo/include/buta_ppo/ffi.h
#pragma once

#include <cstdint>
#include <string>
#include <memory>

// C-ABI definitions matching the Rust exports.
extern "C" {
    bool rs_init(const char* collision_meshes_folder, bool silent);
    bool rs_is_initialized();

    struct rs_arena; // Opaque type representing the Rust Arena struct.
    rs_arena* rs_arena_create(int game_mode_idx);
    void rs_arena_free(rs_arena* arena_ptr);

    void rs_arena_step(rs_arena* arena_ptr);
    uint32_t rs_arena_add_car(rs_arena* arena_ptr, int team_idx);
}

namespace buta_ppo::ffi {

inline bool init(const std::string& collision_meshes_folder, bool silent = true) {
    return rs_init(collision_meshes_folder.c_str(), silent);
}

inline bool is_initialized() {
    return rs_is_initialized();
}

// Let rust handle proper deletion.
struct ArenaDeleter {
    void operator()(rs_arena* ptr) const {
        if (ptr) {
            rs_arena_free(ptr);
        }
    }
};

using ArenaPtr = std::unique_ptr<rs_arena, ArenaDeleter>;

inline ArenaPtr create_arena(int game_mode = 0) {
    return ArenaPtr(rs_arena_create(game_mode));
}

inline void step_arena(const ArenaPtr& arena) {
    if (arena) {
        rs_arena_step(arena.get());
    }
}

enum class Team : int {
    Blue = 0,
    Orange = 1
};

inline uint32_t add_car(const ArenaPtr& arena, Team team = Team::Blue) {
    if (arena) {
        return rs_arena_add_car(arena.get(), static_cast<int>(team));
    }
    return 0; 
}

} // namespace buta_ppo::ffi