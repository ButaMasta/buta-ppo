cmake_minimum_required(VERSION 3.20)
project(buta_ppo LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define directories.
set(RUST_BRIDGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rust_bridge)
set(RUST_TARGET_DIR ${RUST_BRIDGE_DIR}/target/release)
set(RUST_LIB ${RUST_TARGET_DIR}/librust_bridge.so)

# Create a custom target that always runs `cargo build --release`.
add_custom_target(cargo_build ALL
    COMMAND cargo build --release
    WORKING_DIRECTORY ${RUST_BRIDGE_DIR}
    BYPRODUCTS ${RUST_LIB}
    COMMENT "Compiling Rust FFI cdylib via Cargo..."
)

add_compile_options(-Wall -Wextra -O3 -march=native)

# Define the C++ executable.
add_executable(buta_ppo 
    src/main.cpp
    src/env/rocketsim_env.cpp
    src/env/advanced_obs.cpp
    src/env/reward_manager.cpp
)

# Expose include/ directory.
target_include_directories(buta_ppo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link the compiled Rust library.
target_link_libraries(buta_ppo PRIVATE ${RUST_LIB})

# Ensure Cargo builds the Rust library.
add_dependencies(buta_ppo cargo_build)

# (Future) LibTorch Integration Placeholder
# find_package(Torch REQUIRED)
# target_link_libraries(buta_ppo PRIVATE "${TORCH_LIBRARIES}")