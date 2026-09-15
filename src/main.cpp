#include "buta_ppo/ffi.h"
#include <iostream>

int main() {
    std::cout << "Starting RocketSim FFI Initialization..." << std::endl;

    // Initialize RocketSim and load the collision meshes.
    std::string mesh_path = "assets/collision_meshes";
    if (!buta_ppo::ffi::init(mesh_path, false)) {
        std::cerr << "[ERROR] Failed to initialize RocketSim. Check collision mesh path: " 
                  << mesh_path << std::endl;
        return 1;
    }
    
    std::cout << "RocketSim initialized successfully." << std::endl;

    // Spawn test Arena (0 = Soccar).
    std::cout << "Spawning Soccar Arena..." << std::endl;
    auto arena = buta_ppo::ffi::create_arena(0);
    if (!arena) {
        std::cerr << "[ERROR] Arena pointer was null." << std::endl;
        return 1;
    }

    uint32_t blue_car_id = buta_ppo::ffi::add_car(arena, buta_ppo::ffi::Team::Blue);
    uint32_t orange_car_id = buta_ppo::ffi::add_car(arena, buta_ppo::ffi::Team::Orange);

    std::cout << "Spawned Blue Car ID: " << blue_car_id << std::endl;
    std::cout << "Spawned Orange Car ID: " << orange_car_id << std::endl;

    int ticks_to_simulate = 120;
    std::cout << "Stepping arena " << ticks_to_simulate << " times..." << std::endl;

    for (int i = 0; i < ticks_to_simulate; ++i) {
        buta_ppo::ffi::step_arena(arena);
    }

    std::cout << "[SUCCESS] Arena stepped successfully without crashing." << std::endl;

    std::cout << "Shutting down..." << std::endl;
    return 0;
}