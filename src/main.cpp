// buta-ppo/src/main.cpp
#include "buta_ppo/ffi.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

namespace ffi = buta_ppo::ffi;

int main() {
    // Initialize RocketSim and load the collision meshes.
    if (!ffi::init("assets/collision_meshes", true)) {
        std::cerr << "[ERROR] Failed to initialize RocketSim." << std::endl;
        return 1;
    }
    
    auto arena = ffi::create_arena(0);
    if (!arena) return 1;

    ffi::add_car(arena, ffi::Team::Blue);
    ffi::add_car(arena, ffi::Team::Blue);
    ffi::add_car(arena, ffi::Team::Blue);
    ffi::add_car(arena, ffi::Team::Blue);
    ffi::add_car(arena, ffi::Team::Orange);
    ffi::add_car(arena, ffi::Team::Orange);
    ffi::add_car(arena, ffi::Team::Orange);
    ffi::add_car(arena, ffi::Team::Orange);

    ffi::ArenaState global_state{};

    for (int i = 0; i < 300; i++) {
        ffi::step_arena(arena);
    }

    const int num_steps = 1'000'000;
    double dummy_sum = 0.0;

    // Only sim steps with no data fetching.
    std::cout << "\nTest 1: Only stepping the sim with no data fetching. Stepping " << num_steps << " times..." << std::endl;

    auto start_time_1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_steps; i++) {
        ffi::step_arena(arena);
    }
    auto end_time_1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration_1 = end_time_1 - start_time_1;
    double sps_1 = num_steps / duration_1.count();

    std::cout << "Test 1 Completed in " << std::fixed << std::setprecision(4) << duration_1.count() << " seconds." << std::endl;
    std::cout << "-> Steps Per Second: " << std::setprecision(0) << sps_1 << std::endl;

    // Sim steps and data fetching.
    std::cout << "\nTest 2: Stepping the sim and fetching data. Stepping " << num_steps << " times..." << std::endl;

    auto start_time_2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_steps; i++) {
        ffi::step_arena(arena);

        ffi::get_global_state(arena, global_state);

        dummy_sum += global_state.ball.phys.pos[2];
    }
    auto end_time_2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration_2 = end_time_2 - start_time_2;
    double sps_2 = num_steps / duration_2.count();

    std::cout << "Test 2 Completed in " << std::fixed << std::setprecision(4) << duration_2.count() << " seconds." << "Dummy sum: " << dummy_sum << std::endl;
    std::cout << "-> Steps Per Second: " << std::setprecision(0) << sps_2 << std::endl;

    // Review.
    double overhead_pct = ((duration_2.count() - duration_1.count()) / duration_1.count()) * 100.0;

    std::cout << "\n--- Benchmark Results ---" << std::endl;
    std::cout << "FFI Execution and data copying overhead: " << std::setprecision(2) << overhead_pct << "% time increase per tick." << std::endl;

    return 0;
}