// buta-ppo/src/main.cpp
#include "buta_ppo/ffi.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <thread>

namespace ffi = buta_ppo::ffi;

int main() {
    if (!ffi::init("assets/collision_meshes", true)) return 1;
    
    auto arena = ffi::create_arena_vis(0);
    uint32_t blue_car = ffi::add_car(arena, ffi::Team::Blue);
    uint32_t orange_car = ffi::add_car(arena, ffi::Team::Orange);

    // Initial kickoff placement.
    ffi::reset_to_random_kickoff(arena);

    ffi::ArenaState state{};
    int tick = 0;

    while (true) {
        if (tick % 360 == 0) {
            ffi::BallState ball{};
            ball.phys.pos[0] = 0.0f;
            ball.phys.pos[1] = 0.0f;
            ball.phys.pos[2] = 200.0f;
            ball.phys.vel[2] = 1200.0f;
            ball.phys.rot_mat[0][0] = 1.0f;
            ball.phys.rot_mat[1][1] = 1.0f;
            ball.phys.rot_mat[2][2] = 1.0f;

            ffi::set_ball_state(arena, ball);
            std::cout << "[RESET] Ball launched upward!" << std::endl;
        }

        if (tick % 600 == 0) {
            ffi::CarState car{};
            car.phys.pos[0] = 1900.0f;
            car.phys.pos[1] = 500.0f;
            car.phys.pos[2] = 200.0f;
            car.phys.rot_mat[0][0] = 1.0f;
            car.phys.rot_mat[1][1] = 1.0f;
            car.phys.rot_mat[2][2] = 1.0f;

            ffi::set_car_state(arena, blue_car, car);
        }

        ffi::step_arena(arena);
        ffi::get_arena_state(arena, state);

        tick++;
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }

    return 0;
}