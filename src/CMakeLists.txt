if(MSVC)
    set(C_WARN_FLAGS /W3)
    set(CUDA_HOST_WARN_FLAGS --compiler-options=/W3)
else()
    set(C_WARN_FLAGS -Wall -Wextra -Wpedantic)
    set(CUDA_HOST_WARN_FLAGS --compiler-options=-Wall,-Wextra)
endif()

add_library(transpose STATIC
    transpose.cu
    transpose.h
)

target_include_directories(transpose
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(transpose PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(transpose
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_WARN_FLAGS}>
)

# nvcomp — find_package(nvcomp) is broken on Nix (split outputs),
# so locate headers and library manually.
find_path(NVCOMP_INCLUDE_DIR NAMES nvcomp.h
    PATH_SUFFIXES include REQUIRED)
find_library(NVCOMP_LIBRARY NAMES nvcomp_static nvcomp
    PATH_SUFFIXES lib lib/${CUDAToolkit_VERSION_MAJOR} REQUIRED)

add_library(nvcomp::nvcomp UNKNOWN IMPORTED)
set_target_properties(nvcomp::nvcomp PROPERTIES
    IMPORTED_LOCATION "${NVCOMP_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NVCOMP_INCLUDE_DIR}"
)
if(NVCOMP_LIBRARY MATCHES "nvcomp_static")
    set_property(TARGET nvcomp::nvcomp PROPERTY
        INTERFACE_COMPILE_DEFINITIONS NVCOMP_STATIC_DEFINE)
endif()

add_library(chucky_log STATIC
    log/log.c
    log/log.h
)

target_include_directories(chucky_log
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(chucky_log PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(chucky_log PRIVATE _CRT_SECURE_NO_WARNINGS)

target_compile_options(chucky_log
    PRIVATE
        $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>
)

add_library(compress STATIC
    compress.cu
    compress.h
)

target_include_directories(compress
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(compress
    PUBLIC
        nvcomp::nvcomp
        CUDA::cudart_static
        chucky_log
)

set_target_properties(compress PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(compress
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_WARN_FLAGS}>
)

add_library(downsample STATIC
    downsample.cu
    downsample.h
)

target_include_directories(downsample
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(downsample
    PUBLIC
        CUDA::cudart_static
        CUDA::cuda_driver
        chucky_log
)

set_target_properties(downsample PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(downsample
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_WARN_FLAGS}>
)

add_library(aggregate STATIC
    aggregate.cu
    aggregate.h
)

target_include_directories(aggregate
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(aggregate
    PUBLIC
        CUDA::cudart_static
        CUDA::cuda_driver
        chucky_log
)

set_target_properties(aggregate PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(aggregate
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_WARN_FLAGS}>
)

add_library(index_ops STATIC
    index.ops.c
    index.ops.h
)

target_include_directories(index_ops
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(index_ops PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(index_ops
    PRIVATE
        $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>
)

add_library(lod_plan STATIC
    lod_plan.c
    lod_plan.h
)

target_include_directories(lod_plan
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(lod_plan PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(lod_plan
    PUBLIC
        index_ops
)

target_compile_options(lod_plan
    PRIVATE
        $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>
)

add_library(lod STATIC
    lod.cu
    lod.h
)

target_include_directories(lod
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(lod
    PUBLIC
        CUDA::cudart_static
        CUDA::cuda_driver
        chucky_log
)

set_target_properties(lod PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(lod
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_WARN_FLAGS}>
)

if(WIN32)
    set(PLATFORM_SRC platform.win32.c)
else()
    set(PLATFORM_SRC platform.posix.c)
endif()

add_library(stream STATIC
    stream.c
    stream.h
    ${PLATFORM_SRC}
    platform.h
)

target_include_directories(stream
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(stream
    PUBLIC
        transpose
        compress
        aggregate
        lod
        lod_plan
        chucky_log
        CUDA::cuda_driver
)

set_target_properties(stream PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(stream
    PRIVATE
        $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>
)

# JSON writer — zero-allocation snprintf-based JSON serializer
add_library(json_writer STATIC
    json_writer.c
    json_writer.h
)

target_include_directories(json_writer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(json_writer PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(json_writer
    PRIVATE
        $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>
)

# I/O queue — single-thread async job queue
find_package(Threads REQUIRED)

if(WIN32)
    set(IO_QUEUE_SRC io_queue.win32.c)
else()
    set(IO_QUEUE_SRC io_queue.posix.c)
endif()

add_library(io_queue STATIC
    ${IO_QUEUE_SRC}
    io_queue.h
)

target_include_directories(io_queue
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(io_queue
    PUBLIC
        Threads::Threads
        chucky_log
)

set_target_properties(io_queue PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(io_queue
    PRIVATE
        $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>
)

# Platform I/O — cross-platform file operations
if(WIN32)
    set(PLATFORM_IO_SRC platform_io.win32.c)
else()
    set(PLATFORM_IO_SRC platform_io.posix.c)
endif()

add_library(platform_io STATIC
    ${PLATFORM_IO_SRC}
    platform_io.h
)

target_include_directories(platform_io
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(platform_io PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(platform_io
    PRIVATE
        $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>
)

# Zarr v3 store sink
add_library(zarr_sink STATIC
    zarr_sink.c
    zarr_sink.h
)

target_include_directories(zarr_sink
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(zarr_sink
    PUBLIC
        json_writer
        io_queue
        platform_io
        lod_plan
        chucky_log
)

set_target_properties(zarr_sink PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(zarr_sink
    PRIVATE
        $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>
)
