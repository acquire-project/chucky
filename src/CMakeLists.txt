# add_src_lib(TARGET SOURCES src... [LINKS lib...])
#   Creates a STATIC library, exposes its headers, and links warnings.
#   LINKS are added as PUBLIC dependencies.
function(add_src_lib TARGET)
    cmake_parse_arguments(ARG "" "" "SOURCES;LINKS" ${ARGN})
    add_library(${TARGET} STATIC ${ARG_SOURCES})
    target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${TARGET} PRIVATE warnings)
    if(ARG_LINKS)
        target_link_libraries(${TARGET} PUBLIC ${ARG_LINKS})
    endif()
endfunction()

add_src_lib(transpose SOURCES transpose.cu transpose.h)

add_src_lib(chucky_log SOURCES log/log.c log/log.h)

add_src_lib(compress    SOURCES compress.cu compress.h      LINKS nvcomp::nvcomp CUDA::cudart_static chucky_log)
add_src_lib(aggregate   SOURCES aggregate.cu aggregate.h    LINKS CUDA::cudart_static CUDA::cuda_driver chucky_log)
add_src_lib(index_ops   SOURCES index.ops.c index.ops.h)
add_src_lib(lod_plan    SOURCES lod_plan.c lod_plan.h       LINKS index_ops)
add_src_lib(lod         SOURCES lod.cu lod.h                LINKS CUDA::cudart_static CUDA::cuda_driver chucky_log)

add_src_lib(platform SOURCES platform.h)
add_platform_sources(platform)

add_src_lib(crc32c SOURCES crc32c.c crc32c.h)

add_src_lib(writer SOURCES writer.c writer.h LINKS platform io_queue chucky_log)

add_src_lib(shard_delivery SOURCES shard_delivery.c shard_delivery.h
    LINKS crc32c writer aggregate platform chucky_log)

add_src_lib(stream SOURCES stream.c stream.h dimension.c dimension.h
    LINKS transpose compress aggregate lod lod_plan index_ops crc32c shard_delivery writer platform chucky_log CUDA::cuda_driver)

add_src_lib(json_writer SOURCES json_writer.c json_writer.h)

add_src_lib(io_queue SOURCES io_queue.h LINKS Threads::Threads chucky_log)
add_platform_sources(io_queue)

add_src_lib(platform_io SOURCES platform_io.h)
add_platform_sources(platform_io)

add_src_lib(zarr_sink SOURCES zarr_sink.c zarr_sink.h
    LINKS stream json_writer io_queue platform_io lod_plan chucky_log)
