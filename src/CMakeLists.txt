# add_src_lib(TARGET SOURCES src... [LINKS lib...])
#   Creates a STATIC library, exposes its headers, and links warnings.
#   LINKS are added as PUBLIC dependencies.
#   All targets get src/ as their single public include root.
function(add_src_lib TARGET)
    cmake_parse_arguments(ARG "" "" "SOURCES;LINKS" ${ARGN})
    add_library(${TARGET} STATIC ${ARG_SOURCES})
    target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${TARGET} PRIVATE warnings)
    if(ARG_LINKS)
        target_link_libraries(${TARGET} PUBLIC ${ARG_LINKS})
    endif()
endfunction()

# --- Shared libraries (no CUDA dependency) ---

add_src_lib(chucky_log SOURCES log/log.c log/log.h)
add_src_lib(index_ops  SOURCES util/index.ops.c util/index.ops.h)
add_src_lib(lod_plan   SOURCES lod/lod_plan.c lod/lod_plan.h LINKS index_ops)
add_src_lib(crc32c     SOURCES zarr/crc32c.c zarr/crc32c.h)
add_src_lib(dimension  SOURCES dimension.c dimension.h)

add_src_lib(platform SOURCES platform/platform.h)
add_platform_sources(platform platform/platform)

add_src_lib(io_queue SOURCES zarr/io_queue.h LINKS Threads::Threads chucky_log)
add_platform_sources(io_queue zarr/io_queue)

add_src_lib(platform_io SOURCES platform/platform_io.h)
add_platform_sources(platform_io platform/platform_io)

add_src_lib(platform_cmd SOURCES platform/platform_cmd.h)
add_platform_sources(platform_cmd platform/platform_cmd)

add_src_lib(writer SOURCES writer.c writer.h LINKS platform io_queue chucky_log)

add_src_lib(stream_config SOURCES stream/config.c stream/config.h stream/layouts.h
    stream/types.aggregate.c types.stream.h stream/types.aggregate.h types.codec.h types.lod.h defs.limits.h
    LINKS lod_plan index_ops chucky_log)

# --- Backend subdirectories ---

add_subdirectory(gpu)
add_subdirectory(cpu)

add_subdirectory(multiarray)

add_subdirectory(zarr)
