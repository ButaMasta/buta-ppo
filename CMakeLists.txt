cmake_minimum_required(VERSION 3.20)
project(buta_ppo LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_RUST_BRIDGE "Build the Rust bridge via Cargo" ON)

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
        src/util/event_file_writer.cpp
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
    )

    # Release
    target_compile_options(buta_ppo PRIVATE 
        -Wall -Wextra -O3 -march=native -pthread
    )
    # Debug
    # target_compile_options(buta_ppo PRIVATE 
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

# Only LibTorch's CPU libraries. On Windows these are the import libraries of its DLLs.
if(WIN32)
    target_link_libraries(buta_ppo_bot PRIVATE
        ${LIBTORCH_DIR}/lib/torch_cpu.lib
        ${LIBTORCH_DIR}/lib/c10.lib
    )
else()
    target_link_libraries(buta_ppo_bot PRIVATE
        ${LIBTORCH_DIR}/lib/libtorch_cpu.so
        ${LIBTORCH_DIR}/lib/libc10.so
    )
endif()

# The library links its own copy of the Rust bridge, as training does, so its code can call any bridge function. The
# bot binary has a separate copy for its own use. Built here by cargo_build, or passed in by the bot's build.rs (the
# import library rust_bridge.dll.lib on Windows).
if(BUILD_RUST_BRIDGE)
    target_link_libraries(buta_ppo_bot PRIVATE ${RUST_LIB})
    add_dependencies(buta_ppo_bot cargo_build)
else()
    if(NOT RUST_BRIDGE_LIB)
        message(FATAL_ERROR "Set RUST_BRIDGE_LIB to the Rust bridge library to link the bot library against.")
    endif()
    target_link_libraries(buta_ppo_bot PRIVATE ${RUST_BRIDGE_LIB})
endif()

set_target_properties(buta_ppo_bot PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
    # Only the bot API (marked BOT_API in bot_api.cpp) is exported.
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    # LibTorch's DLLs use the DLL C runtime, and C++ objects pass between them and this library, so it must use the
    # same one. Set here because a toolchain file can change the default (cargo-xwin's builds the static one).
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
)

# Release. No -march=native, so the bot runs on other CPUs too.
if(MSVC)
    target_compile_options(buta_ppo_bot PRIVATE
        /W3 /O2 /EHsc /utf-8 /bigobj
    )
    # Debug
    # target_compile_options(buta_ppo_bot PRIVATE
    #     /W3 /Od /Zi /EHsc /utf-8 /bigobj
    # )
else()
    target_compile_options(buta_ppo_bot PRIVATE
        -Wall -Wextra -O3 -pthread
    )
    # Debug
    # target_compile_options(buta_ppo_bot PRIVATE
    #     -Wall -Wextra -g -O0
    # )
endif()
