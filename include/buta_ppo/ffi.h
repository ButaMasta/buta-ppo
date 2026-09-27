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

    struct rs_ball_sim_phys_state {
        float pos[3];
        float vel[3];
        float ang_vel[3];
    };

    struct rs_ball_state {
        rs_phys_state phys;
    };

    struct rs_ball_sim_ball_state {
        rs_ball_sim_phys_state phys;
    };

    struct rs_car_state {
        rs_phys_state phys;
        int32_t team;
        float boost;
        float handbreak_val;
        float air_time_since_jump;
        float demo_respawn_timer;
        bool is_on_ground;
        bool has_jumped;
        bool has_double_jumped;
        bool is_jumping;
        bool has_flip_or_jump;
        bool has_flipped;
        bool is_auto_flipping;
        bool is_flipping;
        bool is_demoed;
        bool is_supersonic;
    };

    struct rs_boost_pad_state {
        float big[6];
        float small[28];
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
        rs_boost_pad_state boost_pads;
        uint32_t num_cars;
        uint64_t tick_count;
        rs_arena_events events;
    };

    struct rs_car_controls {
        float throttle;
        float steer;
        float pitch;
        float yaw;
        float roll;
        bool jump;
        bool boost;
        bool handbrake;
    };

    // Mesh Initialization.
    bool rs_init(const char* collision_meshes_folder, bool silent);
    bool rs_init_ball_sim(const char* collision_meshes_folder, bool silent);

    // Arena Creation.
    struct rs_arena; // Opaque type representing the Rust Arena struct.
    rs_arena* rs_arena_create(int game_mode_idx);
    rs_arena* rs_arena_create_ball_sim(int ball_sim_game_mode_idx);
    rs_arena* rs_arena_create_vis(int game_mode_idx);
    void rs_arena_free(rs_arena* arena_ptr);
    void rs_arena_free_ball_sim(rs_arena* ball_sim_arena_ptr);

    // Arena Util.
    uint32_t rs_arena_add_car(rs_arena* arena_ptr, int team_idx);
    void rs_arena_step(rs_arena* arena_ptr);
    void rs_arena_step_ball_sim(rs_arena* ball_sim_arena_ptr, uint8_t ticks_to_step);
    void rs_arena_get_arena_state(const rs_arena* arena_ptr, rs_arena_state* out_state);
    void rs_ball_sim_arena_get_ball_state(const rs_arena* arena_ptr, rs_ball_sim_ball_state* out_state);
    void rs_arena_set_car_controls(rs_arena* arena_ptr, uint32_t car_idx, rs_car_controls controls);
    void rs_arena_set_ball_state(rs_arena* arena_ptr, rs_ball_state ball_state);
    void rs_arena_set_ball_state_ball_sim(rs_arena* ball_sim_arena_ptr, rs_ball_sim_ball_state ball_state);
    void rs_arena_set_car_state(rs_arena* arena_ptr, uint32_t car_idx, rs_car_state car_state);
    void rs_arena_reset_to_random_kickoff(rs_arena* arena_ptr, uint64_t seed, bool use_seed);
}

namespace buta_ppo::ffi {

using PhysState = rs_phys_state;
using BallSimPhysState = rs_ball_sim_phys_state;
using BallState = rs_ball_state;
using BallSimBallState = rs_ball_sim_ball_state;
using CarState = rs_car_state;
using BoostPadState = rs_boost_pad_state;
using ArenaEvents = rs_arena_events;
using ArenaState = rs_arena_state;
using CarControls = rs_car_controls;

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

struct BallSimArenaDeleter {
    void operator()(rs_arena* ptr) const {
        if (ptr) {
            rs_arena_free_ball_sim(ptr);
        }
    }
};

using ArenaPtr = std::unique_ptr<rs_arena, ArenaDeleter>;
using BallSimArenaPtr = std::unique_ptr<rs_arena, BallSimArenaDeleter>;

inline ArenaPtr create_arena(int game_mode = 0) {
    return ArenaPtr(rs_arena_create(game_mode));
}

inline BallSimArenaPtr create_arena_ball_sim(int ball_sim_game_mode = 0) {
    return BallSimArenaPtr(rs_arena_create_ball_sim(ball_sim_game_mode));
}

inline ArenaPtr create_arena_vis(int game_mode = 0) {
    return ArenaPtr(rs_arena_create_vis(game_mode));
}

inline bool init(const std::string& collision_meshes_folder, bool silent = true) {
    return rs_init(collision_meshes_folder.c_str(), silent);
}

inline bool init_ball_sim(const std::string& collision_meshes_folder, bool silent = true) {
    return rs_init_ball_sim(collision_meshes_folder.c_str(), silent);
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

inline void step_arena_ball_sim(const BallSimArenaPtr& ball_sim_arena, uint8_t ticks_to_step) {
    if (ball_sim_arena) {
        rs_arena_step_ball_sim(ball_sim_arena.get(), ticks_to_step);
    }
}

inline void get_arena_state(const ArenaPtr& arena, ArenaState& out_state) {
    if (arena) {
        rs_arena_get_arena_state(arena.get(), &out_state);
    }
}

inline void get_ball_sim_arena_ball_state(const BallSimArenaPtr& ball_sim_arena, BallSimBallState& out_ball_state) {
    if (ball_sim_arena) {
        rs_ball_sim_arena_get_ball_state(ball_sim_arena.get(), &out_ball_state);
    }
}

inline void set_car_controls(const ArenaPtr& arena, uint32_t car_idx, const CarControls& controls) {
    if (arena) {
        rs_arena_set_car_controls(arena.get(), car_idx, controls);
    }
}

inline void set_ball_state(const ArenaPtr& arena, const BallState& state) {
    if (arena) {
        rs_arena_set_ball_state(arena.get(), state);
    }
}

inline void set_ball_state_ball_sim(const BallSimArenaPtr& ball_sim_arena, const BallSimBallState& state) {
    if (ball_sim_arena) {
        rs_arena_set_ball_state_ball_sim(ball_sim_arena.get(), state);
    }
}

inline void set_car_state(const ArenaPtr& arena, uint32_t car_idx, const CarState& state) {
    if (arena) {
        rs_arena_set_car_state(arena.get(), car_idx, state);
    }
}

inline void reset_to_random_kickoff(const ArenaPtr& arena, uint64_t seed = 0, bool use_seed = false) {
    if (arena) {
        rs_arena_reset_to_random_kickoff(arena.get(), seed, use_seed);
    }
}


} // namespace buta_ppo::ffi