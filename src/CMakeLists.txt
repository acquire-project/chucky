# add_src_lib(TARGET SOURCES src... [LINKS lib...])
#   Creates a STATIC library, exposes its headers, and links warnings.
#   LINKS are added as PUBLIC dependencies.
#   All targets get src/ as their single public include root.
#   Each target is recorded in CHUCKY_INTERNAL_TARGETS so the aggregate
#   library at the bottom of this file can collect its object files.
set(_SRC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
function(add_src_lib TARGET)
    cmake_parse_arguments(ARG "" "" "SOURCES;LINKS" ${ARGN})
    add_library(${TARGET} STATIC ${ARG_SOURCES})
    target_include_directories(${TARGET} PUBLIC ${_SRC_INCLUDE_DIR})
    target_link_libraries(${TARGET} PRIVATE warnings)
    if(ARG_LINKS)
        target_link_libraries(${TARGET} PUBLIC ${ARG_LINKS})
    endif()
    set_property(GLOBAL APPEND PROPERTY CHUCKY_INTERNAL_TARGETS ${TARGET})
endfunction()

set(_math_lib $<$<NOT:$<PLATFORM_ID:Windows>>:m>)

# --- Shared libraries (no CUDA dependency) ---

add_src_lib(chucky_log SOURCES log/log.c log/log.h chucky_log.h)
add_src_lib(index_ops  SOURCES util/index.ops.c util/index.ops.h)
add_src_lib(format_bytes SOURCES util/format_bytes.c util/format_bytes.h)
add_src_lib(strbuf SOURCES util/strbuf.c util/strbuf.h)
add_src_lib(dtype      SOURCES dtype.c dtype.h)
add_src_lib(dimension  SOURCES dimension.c dimension.h)
add_src_lib(lod_plan   SOURCES lod/lod_plan.c lod/lod_plan.h
                               lod/reduce_csr.c lod/reduce_csr.h
                       LINKS index_ops dimension dtype threadpool
)
add_src_lib(crc32c     SOURCES zarr/crc32c.c zarr/crc32c.h LINKS platform)

add_src_lib(platform SOURCES platform/platform.h
    LINKS Threads::Threads $<$<PLATFORM_ID:Windows>:psapi>
)
add_platform_sources(platform platform/platform)

add_src_lib(threadpool SOURCES threadpool/threadpool.c threadpool/threadpool.h
    LINKS platform
)

add_src_lib(io_queue SOURCES zarr/io_queue.h zarr/types.io.h zarr/io_request.h
    zarr/io_queue_stats.c zarr/io_queue_stats.h
    LINKS Threads::Threads platform chucky_log)
add_platform_sources(io_queue zarr/io_queue)

add_src_lib(platform_io SOURCES platform/platform_io.h)
add_platform_sources(platform_io platform/platform_io)

add_src_lib(platform_cmd SOURCES platform/platform_cmd.h)
add_platform_sources(platform_cmd platform/platform_cmd)

add_src_lib(writer SOURCES writer.c writer.h LINKS platform io_queue chucky_log)

add_src_lib(store_api SOURCES store.c store.h)

add_src_lib(codec_types SOURCES types.codec.c types.codec.h)

add_src_lib(stream_config SOURCES stream/config.c stream/config.h stream/layouts.h
    stream/dim_info.c stream/dim_info.h
    stream/types.aggregate.c types.stream.h stream/types.aggregate.h types.lod.h defs.limits.h
    LINKS ${_math_lib} codec_types lod_plan index_ops platform chucky_log
)

# --- Backend subdirectories ---

if(CHUCKY_ENABLE_GPU)
    add_subdirectory(gpu)
endif()
add_subdirectory(cpu)

add_subdirectory(multiarray)

add_subdirectory(zarr)
add_subdirectory(ngff)
add_subdirectory(hcs)

# --- Aggregate library: chucky (GPU+CPU) or chucky-cpu (CPU only) ---
# Merges all internal STATIC libs into one combined library. The static
# variant is always built; the shared variant is opt-in (CHUCKY_BUILD_SHARED)
# because it requires every transitive dep to be -fPIC, which is not the
# case for system static libs in the CI Docker images.

option(CHUCKY_BUILD_SHARED "Build aggregate as a shared library too" OFF)

set(CHUCKY_AGGREGATE_NAME chucky-cpu)
if(CHUCKY_ENABLE_GPU)
    set(CHUCKY_AGGREGATE_NAME chucky)
endif()
set(CHUCKY_AGGREGATE_NAME ${CHUCKY_AGGREGATE_NAME} PARENT_SCOPE)

get_property(_chucky_internal GLOBAL PROPERTY CHUCKY_INTERNAL_TARGETS)
list(
    TRANSFORM _chucky_internal
    REPLACE "(.+)" "$<TARGET_OBJECTS:\\1>"
    OUTPUT_VARIABLE _chucky_objects
)

# Conditional deps via generator expressions — empty branches are dropped at
# generation time without referencing missing targets.
set(_with_blosc $<BOOL:${HAVE_BLOSC}>)
set(_with_gpu $<BOOL:${CHUCKY_ENABLE_GPU}>)
set(_chucky_public_deps
    ${_math_lib}
    Threads::Threads
    Lz4::Lz4
    Zstd::Zstd
    AWS::aws-c-s3
    $<$<PLATFORM_ID:Windows>:psapi>
    $<${_with_blosc}:Blosc::Blosc>
    $<${_with_gpu}:CUDA::cuda_driver>
    $<${_with_gpu}:CUDA::cudart_static>
    $<${_with_gpu}:nvcomp::nvcomp>
)

set(_chucky_kinds static)
if(CHUCKY_BUILD_SHARED)
    list(APPEND _chucky_kinds shared)
endif()
foreach(_kind IN LISTS _chucky_kinds)
    string(TOUPPER ${_kind} _KIND)
    add_library(${CHUCKY_AGGREGATE_NAME}-${_kind} ${_KIND} ${_chucky_objects})
    set_target_properties(
        ${CHUCKY_AGGREGATE_NAME}-${_kind}
        PROPERTIES OUTPUT_NAME ${CHUCKY_AGGREGATE_NAME}
    )
    target_include_directories(
        ${CHUCKY_AGGREGATE_NAME}-${_kind}
        INTERFACE
            $<BUILD_INTERFACE:${_SRC_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(
        ${CHUCKY_AGGREGATE_NAME}-${_kind}
        PUBLIC ${_chucky_public_deps}
    )
endforeach()
