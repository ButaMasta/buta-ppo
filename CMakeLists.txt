cmake_minimum_required(VERSION 3.20)
project(buta_ppo LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_RUST_BRIDGE "Build the Rust bridge via Cargo" ON)

# Grab dependency for logging.
find_package(Protobuf REQUIRED)
find_package(absl REQUIRED)
include(FetchContent)
FetchContent_Declare(
    tensorboard_logger
    GIT_REPOSITORY      https://github.com/RustingSword/tensorboard_logger.git
    GIT_TAG             master
)
FetchContent_MakeAvailable(tensorboard_logger)

# Find the Torch package using the provided path
set(Torch_DIR "/home/buta/Dev/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(BUILD_RUST_BRIDGE)

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

    # Define the C++ executable.
    add_executable(buta_ppo 
        src/main.cpp
        src/action/default_action.cpp
        src/env/rocketsim_env.cpp
        src/env/vec_env.cpp
        src/obs/advanced_obs.cpp
        src/reward/reward_manager.cpp
        src/reward/event_goal.cpp
        src/reward/touch_ball.cpp
        src/reward/velocity_to_ball.cpp
        src/rl/rollout_buffer.cpp
        src/rl/actor_critic.cpp
        src/rl/ppo_trainer.cpp
        src/rl/ppo_runner.cpp
        src/state/default_kickoff.cpp
        src/state/random_state.cpp
    )

    # Expose include/ directory.
    target_include_directories(buta_ppo PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # Link the compiled Rust library.
    target_link_libraries(buta_ppo PRIVATE 
        ${RUST_LIB} 
        ${TORCH_LIBRARIES} 
        tensorboard_logger
        protobuf::libprotobuf
        absl::log_internal_check_op
    )

    # Release
    target_compile_options(buta_ppo PRIVATE 
        "-I${Protobuf_INCLUDE_DIRS}"
        -Wall -Wextra -O3 -march=native -pthread
    )
    # Debug
    # target_compile_options(buta_ppo PRIVATE 
    #     "-I${Protobuf_INCLUDE_DIRS}"
    #     -Wall -Wextra -g -O0
    # )

    # Ensure Cargo builds the Rust library.
    add_dependencies(buta_ppo cargo_build)

endif()

# Bot target.
add_library(buta_ppo_bot SHARED
    src/bot_api.cpp
    src/action/default_action.cpp
    src/obs/advanced_obs.cpp
    src/rl/actor_critic.cpp
)

target_include_directories(buta_ppo_bot PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(buta_ppo_bot PRIVATE
    ${TORCH_LIBRARIES}
)

# Release
target_compile_options(buta_ppo_bot PRIVATE 
    -Wall -Wextra -O3 -march=native -pthread
)
# Debug
# target_compile_options(buta_ppo_bot PRIVATE 
#     -Wall -Wextra -g -O0
# )