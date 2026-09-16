// buta-ppo/include/buta_ppo/ffi.h
#pragma once

#include <cstdint>
#include <string>
#include <memory>

// C-ABI definitions matching the Rust exports.
extern "C" {
    // C++ State Interface Structs.
    struct rs_phys_state {
        float pos[3];
        float rot_mat[3][3];
        float vel[3];
        float ang_vel[3];
    };

    struct rs_ball_state {
        rs_phys_state phys;
    };

    struct rs_car_state {
        rs_phys_state phys;
        int32_t team;
        float boost;
        float handbreak_val;
        bool is_on_ground;
        bool has_jumped;
        bool has_double_jumped;
        bool is_jumping;
        bool has_flipped;
        bool is_flipping;
        bool is_demoed;
        bool is_supersonic;
    };

    struct rs_arena_events {
        bool is_ball_scored;
        bool car_hit_ball[8];
        bool car_got_boost[8];
        float ball_hit_extra_vel[8];
    };

    struct rs_arena_state {
        rs_ball_state ball;
        rs_car_state cars[8];
        uint32_t num_cars;
        uint64_t tick_count;
        rs_arena_events events;
    };

    // Mesh Initialization.
    bool rs_init(const char* collision_meshes_folder, bool silent);

    // Arena Creation.
    struct rs_arena; // Opaque type representing the Rust Arena struct.
    rs_arena* rs_arena_create(int game_mode_idx);
    void rs_arena_free(rs_arena* arena_ptr);

    // Arena Util.
    uint32_t rs_arena_add_car(rs_arena* arena_ptr, int team_idx);
    void rs_arena_step(rs_arena* arena_ptr);
    void rs_arena_get_global_state(const rs_arena* arena_ptr, rs_arena_state* out_state);
}

namespace buta_ppo::ffi {

using PhysState = rs_phys_state;
using BallState = rs_ball_state;
using CarState = rs_car_state;
using ArenaEvents = rs_arena_events;
using ArenaState = rs_arena_state;

enum class Team : int {
    Blue = 0,
    Orange = 1
};
    

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

inline bool init(const std::string& collision_meshes_folder, bool silent = true) {
    return rs_init(collision_meshes_folder.c_str(), silent);
}

inline uint32_t add_car(const ArenaPtr& arena, Team team = Team::Blue) {
    if (arena) {
        return rs_arena_add_car(arena.get(), static_cast<int>(team));
    }
    return 0; 
}

inline void step_arena(const ArenaPtr& arena) {
    if (arena) {
        rs_arena_step(arena.get());
    }
}

inline void get_global_state(const ArenaPtr& arena, ArenaState& out_state) {
    if (arena) {
        rs_arena_get_global_state(arena.get(), &out_state);
    }
}


} // namespace buta_ppo::ffi