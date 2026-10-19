# Widens tensorboard_logger's public step parameters from int to int64_t. Upstream truncates the step to 32 bits,
# so steps past 2^31 log as negative. The event format and the private add_event already use int64_t.
foreach(file include/tensorboard_logger.h src/tensorboard_logger.cc)
    file(READ "${file}" contents)
    string(REPLACE "int step" "int64_t step" contents "${contents}")

    # buta-ppo logs through add_scalar, so it has to come out 64-bit. If upstream changes its signature, the
    # replacement above can silently miss it, and steps past 2^31 would log as negative again.
    if(NOT contents MATCHES "add_scalar\\([^)]*int64_t step")
        message(FATAL_ERROR "patch_tensorboard_logger.cmake: add_scalar in ${file} doesn't take an int64_t step after "
                            "patching. The upstream signature changed, so update this patch for the pinned GIT_TAG.")
    endif()

    file(WRITE "${file}" "${contents}")
endforeach()
