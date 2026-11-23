# buta-ppo/cmake/libtorch_checks.cmake
# Configure-time checks for setup mistakes that would otherwise show up late or confusingly: as a link error, a missing
# DLL at startup, or LibTorch's own generic "cannot find the CUDA libraries". CMakeLists.txt includes this once
# LIBTORCH_DIR is known, calls the checks, then finds LibTorch with buta_ppo_find_libtorch().

# The LibTorch version, e.g. "2.15.0+cu134", from its build-version file, or "" if it has none.
function(buta_ppo_libtorch_version out)
    set(version "")
    if(EXISTS "${LIBTORCH_DIR}/build-version")
        file(STRINGS "${LIBTORCH_DIR}/build-version" version LIMIT_COUNT 1)
    endif()
    set(${out} "${version}" PARENT_SCOPE)
endfunction()

# Windows: LibTorch's Debug download uses the debug C runtime, which can't be mixed with the release runtime the targets
# here use (MSVC_RUNTIME_LIBRARY in CMakeLists.txt), so linking would fail with "mismatch detected" errors. Its DLLs
# import the debug runtime's DLLs (MSVCP140D.dll, VCRUNTIME140D.dll, ucrtbased.dll), which is how it's recognized.
function(buta_ppo_check_libtorch_release)
    if(NOT WIN32 OR NOT EXISTS "${LIBTORCH_DIR}/lib/c10.dll")
        return()
    endif()
    file(STRINGS "${LIBTORCH_DIR}/lib/c10.dll" debug_runtime LIMIT_COUNT 1
         REGEX "^(MSVCP|VCRUNTIME)140[Dd]\\.[Dd][Ll][Ll]$|^[Uu][Cc][Rr][Tt][Bb][Aa][Ss][Ee][Dd]\\.[Dd][Ll][Ll]$")
    if(debug_runtime)
        message(FATAL_ERROR "LibTorch at '${LIBTORCH_DIR}' is the Debug build (its DLLs use the debug C runtime, "
                            "${debug_runtime}), which this project can't link with. Download the Release build "
                            "instead: libtorch-win-shared-with-deps-<version>.zip, not the -debug- one.")
    endif()
endfunction()

# The training build: a single-configuration generator, a CUDA build of LibTorch, and a CUDA Toolkit of the same major
# version as LibTorch's CUDA.
function(buta_ppo_check_training_setup)
    # Multi-configuration generators (Visual Studio, CMake's default on Windows) build Debug unless told otherwise, and
    # put the executable in a per-configuration subdirectory. The training build has one configuration with its own
    # optimization flags, and copies LibTorch's DLLs next to the executable in the build directory.
    get_property(multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(multi_config)
        message(FATAL_ERROR "The training build doesn't support the '${CMAKE_GENERATOR}' generator. Delete the build "
                            "directory, then configure with Ninja: cmake -G Ninja -B build (on Windows, from a "
                            "Developer PowerShell for Visual Studio, which has Ninja).")
    endif()

    buta_ppo_libtorch_version(version)
    set(described "LibTorch at '${LIBTORCH_DIR}'")
    if(version)
        set(described "${described} (${version})")
    endif()

    # Training runs on an NVIDIA GPU. A CPU build would compile and link, then fail when training starts.
    if(WIN32)
        set(cuda_lib "torch_cuda.lib")
    else()
        set(cuda_lib "libtorch_cuda.so")
    endif()
    if(NOT EXISTS "${LIBTORCH_DIR}/lib/${cuda_lib}")
        message(FATAL_ERROR "${described} is a CPU-only build (it has no lib/${cuda_lib}), but training runs on an "
                            "NVIDIA GPU. Download a CUDA build of LibTorch (a version ending in +cu<CUDA version>, "
                            "such as 2.15.0+cu134), set LIBTORCH to it, and configure again. A CPU build is only for "
                            "building the RLBot bot (see docs/deploying.md).")
    endif()

    # Official downloads say which CUDA they're built for (+cu134 = CUDA 13.4). Without that, leave the CUDA Toolkit to
    # LibTorch's own config.
    if(NOT version MATCHES "\\+cu([0-9]+)([0-9])")
        return()
    endif()
    set(torch_cuda_major "${CMAKE_MATCH_1}")
    set(torch_cuda "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")

    # LibTorch's config needs the CUDA Toolkit (nvcc and the CUDA headers), and looks for it in much the same places:
    # CUDA_PATH (set by NVIDIA's Windows installer), PATH, then the usual Linux install directories.
    find_program(BUTA_PPO_NVCC nvcc
        HINTS ${CUDAToolkit_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} ENV CUDAToolkit_ROOT ENV CUDA_PATH ENV CUDA_HOME
        PATHS /usr/local/cuda /opt/cuda
        PATH_SUFFIXES bin)
    mark_as_advanced(BUTA_PPO_NVCC)
    if(NOT BUTA_PPO_NVCC)
        message(FATAL_ERROR "${described} is built for CUDA ${torch_cuda}, so building against it needs the CUDA "
                            "Toolkit ${torch_cuda_major}.x, and none was found. Install it from NVIDIA, or if it's "
                            "installed, put its bin directory on PATH or set CUDA_PATH to its directory, then "
                            "configure again.")
    endif()

    # The program links the toolkit's CUDA runtime, and LibTorch's DLLs bring their own. Within a major version one
    # serves both, but across major versions the program would load two runtimes, or fail to find the toolkit's.
    execute_process(COMMAND "${BUTA_PPO_NVCC}" --version OUTPUT_VARIABLE nvcc_output ERROR_QUIET)
    if(nvcc_output MATCHES "release ([0-9]+)\\.([0-9]+)")
        if(NOT CMAKE_MATCH_1 STREQUAL torch_cuda_major)
            message(FATAL_ERROR "${described} is built for CUDA ${torch_cuda}, but the CUDA Toolkit found is "
                                "${CMAKE_MATCH_1}.${CMAKE_MATCH_2} (${BUTA_PPO_NVCC}). Their major versions must "
                                "match: install CUDA Toolkit ${torch_cuda_major}.x and make it the one found (PATH or "
                                "CUDA_PATH), or download the LibTorch build for your toolkit's CUDA version.")
        endif()
    endif()
endfunction()

# The cached library paths (FILEPATH entries named *_LIBRARY) that name a library LIBTORCH_DIR/lib has, but in another
# directory: another LibTorch's copy.
function(buta_ppo_other_libtorch_libraries out)
    file(REAL_PATH "${LIBTORCH_DIR}/lib" own_dir)
    if(WIN32)
        string(TOLOWER "${own_dir}" own_dir)
    endif()
    set(others "")
    get_property(cache_vars DIRECTORY PROPERTY CACHE_VARIABLES)
    foreach(var IN LISTS cache_vars)
        # Filtered before reading any value: reading some variables (CMAKE_WARN_DEPRECATED) makes CMake 4.4+ warn.
        if(NOT var MATCHES "_LIBRARY$")
            continue()
        endif()
        get_property(type CACHE "${var}" PROPERTY TYPE)
        get_property(path CACHE "${var}" PROPERTY VALUE)
        if(NOT type STREQUAL "FILEPATH" OR NOT EXISTS "${path}")
            continue()
        endif()
        get_filename_component(name "${path}" NAME)
        get_filename_component(dir "${path}" DIRECTORY)
        file(REAL_PATH "${dir}" dir)
        if(WIN32)
            string(TOLOWER "${dir}" dir)
        endif()
        if(EXISTS "${LIBTORCH_DIR}/lib/${name}" AND NOT dir STREQUAL own_dir)
            list(APPEND others "${var}")
        endif()
    endforeach()
    set(${out} "${others}" PARENT_SCOPE)
endfunction()

# find_package(Torch), making sure every library comes from LIBTORCH_DIR. LibTorch's config looks for its libraries
# with find_library(... PATHS <its lib/>), which CMake searches last, after PATH (on Windows) and the system library
# directories, so another LibTorch there would be linked silently, with this one's headers. And find_library caches what
# it finds, so a build directory configured with an earlier LIBTORCH would keep linking that one's libraries.
macro(buta_ppo_find_libtorch)
    # Forget the libraries an earlier LIBTORCH left in the cache, so they're looked up again.
    buta_ppo_other_libtorch_libraries(_buta_ppo_stale)
    foreach(_buta_ppo_var IN LISTS _buta_ppo_stale)
        unset(${_buta_ppo_var} CACHE)
    endforeach()

    # CMAKE_LIBRARY_PATH is searched before PATH and the system directories. It only affects find_library, and
    # LibTorch's lib/ has no library named like a CUDA Toolkit one (it bundles those under versioned names only), so the
    # CUDA lookups inside LibTorch's config find what they found before.
    set(_buta_ppo_library_path "${CMAKE_LIBRARY_PATH}")
    set(CMAKE_LIBRARY_PATH "${LIBTORCH_DIR}/lib" ${CMAKE_LIBRARY_PATH})
    set(Torch_DIR "${LIBTORCH_DIR}/share/cmake/Torch")
    find_package(Torch REQUIRED)
    set(CMAKE_LIBRARY_PATH "${_buta_ppo_library_path}")

    buta_ppo_other_libtorch_libraries(_buta_ppo_other)
    if(_buta_ppo_other)
        set(_buta_ppo_found "")
        foreach(_buta_ppo_var IN LISTS _buta_ppo_other)
            string(APPEND _buta_ppo_found "\n  ${_buta_ppo_var} = ${${_buta_ppo_var}}")
        endforeach()
        message(FATAL_ERROR "LibTorch's libraries must come from LIBTORCH ('${LIBTORCH_DIR}/lib'), but these were "
                            "found elsewhere, in another LibTorch:${_buta_ppo_found}\nRemove that LibTorch's directory "
                            "from PATH (and LIB), then configure again.")
    endif()
endmacro()
