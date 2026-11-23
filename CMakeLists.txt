cmake_minimum_required(VERSION 3.20)

# No build type unless one is given: the targets set their own flags (for a debug build, swap their commented blocks).
# Otherwise project() would default MSVC to Debug, whose /Od and /RTC1 clash with the targets' /O2.
set(CMAKE_BUILD_TYPE "" CACHE STRING "Leave empty: the targets set their own optimization flags.")

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

# Catch the common LibTorch setup mistakes with a clear message, before LibTorch's own config reports them confusingly.
include(cmake/libtorch_checks.cmake)
buta_ppo_check_libtorch_release()
if(BUILD_RUST_BRIDGE)
    buta_ppo_check_training_setup()
endif()

buta_ppo_find_libtorch()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(BUILD_RUST_BRIDGE)

    # Define directories.
    set(RUST_BRIDGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rust_bridge)
    set(RUST_TARGET_DIR ${RUST_BRIDGE_DIR}/target/release)
    # What to link: on Windows the DLL's import library. RUST_DLL is what runs, copied next to the executable below.
    if(WIN32)
        set(RUST_LIB ${RUST_TARGET_DIR}/rust_bridge.dll.lib)
        set(RUST_DLL ${RUST_TARGET_DIR}/rust_bridge.dll)
    else()
        set(RUST_LIB ${RUST_TARGET_DIR}/librust_bridge.so)
        set(RUST_DLL "")
    endif()

    # Create a custom target that always runs `cargo build --release`. That builds only the bridge library, not the
    # RLBot bot package in the same Cargo workspace, so it's a no-op unless the Rust code changed.
    add_custom_target(cargo_build ALL
        COMMAND cargo build --release
        WORKING_DIRECTORY ${RUST_BRIDGE_DIR}
        BYPRODUCTS ${RUST_LIB} ${RUST_DLL}
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

    # OS-specific code: each src/platform/ header has an implementation per OS.
    if(WIN32)
        target_sources(buta_ppo PRIVATE
            src/platform/cpu_windows.cpp
            src/platform/process_windows.cpp
            src/platform/timing_windows.cpp
        )
    else()
        target_sources(buta_ppo PRIVATE
            src/platform/cpu_linux.cpp
            src/platform/process_linux.cpp
            src/platform/timing_linux.cpp
        )
    endif()

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

    if(MSVC)
        # Release. LibTorch's targets add /permissive- too.
        target_compile_options(buta_ppo PRIVATE
            /W3 /O2 /EHsc /utf-8 /bigobj
        )
        # Debug
        # target_compile_options(buta_ppo PRIVATE
        #     /W3 /Od /Zi /EHsc /utf-8 /bigobj
        # )
        # MSVC deprecates standard C functions such as std::localtime in favor of its own _s variants, which other
        # compilers don't have. The standard ones stay.
        target_compile_definitions(buta_ppo PRIVATE _CRT_SECURE_NO_WARNINGS)
        # MSVC can't tune for the build machine's CPU, so it gets AVX2 (Haswell or newer). clang-cl can, like GCC.
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            target_compile_options(buta_ppo PRIVATE -march=native)
        else()
            target_compile_options(buta_ppo PRIVATE /arch:AVX2)
        endif()
    else()
        # Release
        target_compile_options(buta_ppo PRIVATE
            -Wall -Wextra -O3 -march=native -pthread
        )
        # Debug
        # target_compile_options(buta_ppo PRIVATE
        #     -Wall -Wextra -g -O0
        # )
    endif()

    # The DLL C runtime, as LibTorch's DLLs use, since C++ objects pass between them and the program. Set explicitly
    # because a Debug build type or a toolchain file can change the default.
    set_target_properties(buta_ppo PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

    # Ensure Cargo builds the Rust library.
    add_dependencies(buta_ppo cargo_build)

    # Windows has no rpath: a program finds its DLLs next to itself (or on PATH). So LibTorch's DLLs (about 3 GB for a
    # CUDA build) and the bridge's are copied into the build directory, where Ninja puts buta_ppo.exe. Each is copied
    # again only when its source is newer, so LibTorch's once per LibTorch version.
    if(WIN32)
        file(GLOB TORCH_DLLS CONFIGURE_DEPENDS "${LIBTORCH_DIR}/lib/*.dll")
        set(RUNTIME_DLLS "")
        foreach(dll IN LISTS TORCH_DLLS RUST_DLL)
            get_filename_component(dll_name "${dll}" NAME)
            set(copied "${CMAKE_CURRENT_BINARY_DIR}/${dll_name}")
            add_custom_command(
                OUTPUT "${copied}"
                COMMAND ${CMAKE_COMMAND} -E copy "${dll}" "${copied}"
                DEPENDS "${dll}"
                VERBATIM
            )
            list(APPEND RUNTIME_DLLS "${copied}")
        endforeach()
        add_custom_target(copy_runtime_dlls ALL DEPENDS ${RUNTIME_DLLS})
    endif()

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
