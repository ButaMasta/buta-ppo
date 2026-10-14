# Widens tensorboard_logger's public step parameters from int to int64_t. Upstream truncates the step to 32 bits,
# so steps past 2^31 log as negative. The event format and the private add_event already use int64_t.
foreach(file include/tensorboard_logger.h src/tensorboard_logger.cc)
    file(READ "${file}" contents)
    string(REPLACE "int step" "int64_t step" contents "${contents}")
    file(WRITE "${file}" "${contents}")
endforeach()
