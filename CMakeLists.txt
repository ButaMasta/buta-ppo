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
    GIT_TAG             69267c5782775a97bc9c01bec3fa537b739f79fd
    PATCH_COMMAND       ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_tensorboard_logger.cmake
)
FetchContent_MakeAvailable(tensorboard_logger)

# Find LibTorch from the LIBTORCH environment variable. The path is cached on configure, so later builds don't need it set.
if(DEFINED ENV{LIBTORCH})
    set(LIBTORCH_DIR "$ENV{LIBTORCH}" CACHE PATH "LibTorch directory, read from the LIBTORCH environment variable." FORCE)
endif()
if(NOT LIBTORCH_DIR)
    message(FATAL_ERROR "LIBTORCH is not set. Set the LIBTORCH environment variable to your LibTorch directory "
                        "(the one containing lib/ and share/), then configure again.")
endif()
if(NOT EXISTS "${LIBTORCH_DIR}/share/cmake/Torch/TorchConfig.cmake")
    set(bad_dir "${LIBTORCH_DIR}")
    unset(LIBTORCH_DIR CACHE)
    message(FATAL_ERROR "LibTorch not found at '${bad_dir}'. Set the LIBTORCH environment variable to your LibTorch "
                        "directory (the one containing lib/ and share/), then configure again.")
endif()
set(Torch_DIR "${LIBTORCH_DIR}/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(BUILD_RUST_BRIDGE)

    # Define directories.
    set(RUST_BRIDGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rust_bridge)
    set(RUST_TARGET_DIR ${RUST_BRIDGE_DIR}/target/release)
    set(RUST_LIB ${RUST_TARGET_DIR}/librust_bridge.so)

    # Create a custom target that always runs `cargo build --release`. That builds only the bridge library, not the
    # RLBot bot package in the same Cargo workspace, so it's a no-op unless the Rust code changed.
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
        src/env/eval_vec_env.cpp
        src/external/nexto/nexto_bot.cpp
        src/external/nexto/nexto_net.cpp
        src/obs/default_obs.cpp
        src/metric/metric_manager.cpp
        src/metric/in_air_ratio.cpp
        src/reward/reward_manager.cpp
        src/reward/event_goal.cpp
        src/reward/touch_ball.cpp
        src/reward/velocity_to_ball.cpp
        src/reward/velocity_ball_to_goal.cpp
        src/rl/rating.cpp
        src/rl/eval_ledger.cpp
        src/rl/rollout_buffer.cpp
        src/rl/actor_critic.cpp
        src/rl/optimizers.cpp
        src/rl/ppo_trainer.cpp
        src/rl/ppo_runner.cpp
        src/rl/ppo_benchmark.cpp
        src/rl/training_plan.cpp
        src/rl/opponent_pool.cpp
        src/rl/gauntlet.cpp
        src/rl/nexto_ladder.cpp
        src/state/default_kickoff.cpp
        src/state/random_state.cpp
        src/state/eval_kickoff.cpp
        src/util/worker_pool.cpp
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

# Bot target. It runs inference on the CPU, so it links only LibTorch's CPU libraries, and finds them next to itself in
# the bot's libs/ directory (deploy.py copies them there). That keeps the packaged bot runnable from any location and on
# machines without LibTorch or a GPU.
add_library(buta_ppo_bot SHARED
    src/bot_api.cpp
    src/action/default_action.cpp
    src/obs/default_obs.cpp
    src/rl/actor_critic.cpp
)

target_include_directories(buta_ppo_bot PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(buta_ppo_bot SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})

target_link_libraries(buta_ppo_bot PRIVATE
    ${LIBTORCH_DIR}/lib/libtorch_cpu.so
    ${LIBTORCH_DIR}/lib/libc10.so
)

set_target_properties(buta_ppo_bot PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
)

# Release. No -march=native, so the bot runs on other CPUs too.
target_compile_options(buta_ppo_bot PRIVATE 
    -Wall -Wextra -O3 -pthread
)
# Debug
# target_compile_options(buta_ppo_bot PRIVATE 
#     -Wall -Wextra -g -O0
# )