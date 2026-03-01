if(MSVC)
    set(C_WARN_FLAGS /W3)
    set(AVX2_FLAG /arch:AVX2)
else()
    set(C_WARN_FLAGS -Wall -Wextra -Wpedantic)
    set(AVX2_FLAG -mavx2)
endif()

# Helper function to create test executables with common settings
function(add_my_test TGT SOURCE_FILE)
    add_executable(${TGT} ${SOURCE_FILE})

    target_link_libraries(${TGT}
        PRIVATE
            CUDA::cuda_driver
            ${ARGN}
    )

    if(NOT WIN32)
        set_target_properties(${TGT} PROPERTIES
            BUILD_RPATH "/run/opengl-driver/lib"
            INSTALL_RPATH "/run/opengl-driver/lib"
        )
    endif()

    target_compile_options(${TGT}
        PRIVATE
            $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>
    )

    set_target_properties(${TGT} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    
    add_test(NAME "test-${TGT}" COMMAND ${TGT})
endfunction()

add_my_test(test_ctx_null test_ctx_null.c)

if(MSVC)
    set(OPENMP_FLAGS /openmp:llvm)
else()
    find_package(OpenMP REQUIRED COMPONENTS C)
    set(OPENMP_FLAGS ${OpenMP_C_FLAGS})
endif()

# Index operations utility library (shared between scalar and AVX2 tests)
add_library(index_ops_util STATIC index.ops.util.c)
target_compile_options(index_ops_util PRIVATE ${C_WARN_FLAGS})
target_include_directories(index_ops_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(index_ops_util PUBLIC index_ops)

add_my_test(index.ops index.ops.c)
target_compile_options(index.ops PRIVATE ${OPENMP_FLAGS})
target_link_libraries(index.ops PRIVATE index_ops_util stream)

# AVX2 helper library
add_library(div_avx2 STATIC div.avx2.c)
target_compile_options(div_avx2 PRIVATE ${AVX2_FLAG} ${C_WARN_FLAGS})
target_include_directories(div_avx2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# AVX2 test - always build, runtime check will skip if CPU doesn't support
add_my_test(index.ops.avx2 index.ops.avx2.c)
target_compile_options(index.ops.avx2 PRIVATE ${OPENMP_FLAGS} ${AVX2_FLAG})
target_link_libraries(index.ops.avx2 PRIVATE div_avx2 index_ops_util stream)

# Magic division test
add_executable(test_magic_div test_magic_div.c)
target_compile_options(test_magic_div PRIVATE ${C_WARN_FLAGS})
target_link_libraries(test_magic_div PRIVATE div_avx2)

# AVX2 division test
add_executable(test_avx2_div test_avx2_div.c)
target_compile_options(test_avx2_div PRIVATE ${AVX2_FLAG} ${C_WARN_FLAGS})
target_link_libraries(test_avx2_div PRIVATE div_avx2)

# Transpose indices CUDA test
add_my_test(index.ops.cuda index.ops.cuda.c transpose index_ops_util chucky_log CUDA::cudart_static)

# Test platform utilities
if(WIN32)
    set(TEST_PLATFORM_SRC test_platform.win32.c)
else()
    set(TEST_PLATFORM_SRC test_platform.posix.c)
endif()

add_library(test_platform STATIC
    ${TEST_PLATFORM_SRC}
    test_platform.h
)
target_include_directories(test_platform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_platform PRIVATE $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>)

# Shared test data generation library
add_library(test_data STATIC test_data.c test_data.h)
target_include_directories(test_data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_data PUBLIC stream)
target_compile_options(test_data PRIVATE $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>)

# Streaming tiled transpose test
find_path(ZSTD_INCLUDE_DIR NAMES zstd.h
    PATH_SUFFIXES include REQUIRED)
find_library(ZSTD_LIBRARY NAMES zstd libzstd_static zstd_static
    PATH_SUFFIXES lib static REQUIRED)

add_my_test(test_stream test_stream.c stream CUDA::cudart_static)
target_link_libraries(test_stream PRIVATE index_ops_util ${ZSTD_LIBRARY})
target_include_directories(test_stream PRIVATE ${ZSTD_INCLUDE_DIR})

add_my_test(test_compress test_compress.c stream CUDA::cudart_static)
target_link_libraries(test_compress PRIVATE ${ZSTD_LIBRARY})
target_include_directories(test_compress PRIVATE ${ZSTD_INCLUDE_DIR})

add_my_test(test_bench test_bench.c stream CUDA::cudart_static)
target_link_libraries(test_bench PRIVATE test_data ${ZSTD_LIBRARY})
target_include_directories(test_bench PRIVATE ${ZSTD_INCLUDE_DIR})

add_my_test(test_aggregate test_aggregate.c aggregate CUDA::cudart_static)

add_my_test(test_shard_contents test_shard_contents.c stream CUDA::cudart_static)
target_link_libraries(test_shard_contents PRIVATE ${ZSTD_LIBRARY})
target_include_directories(test_shard_contents PRIVATE ${ZSTD_INCLUDE_DIR})

# JSON writer test (no CUDA)
add_executable(test_json_writer test_json_writer.c)
target_link_libraries(test_json_writer PRIVATE json_writer chucky_log)
target_compile_options(test_json_writer PRIVATE ${C_WARN_FLAGS})
add_test(NAME "test-test_json_writer" COMMAND test_json_writer)

# I/O queue test (no CUDA)
add_executable(test_io_queue test_io_queue.c)
target_link_libraries(test_io_queue PRIVATE io_queue chucky_log)
target_compile_options(test_io_queue PRIVATE ${C_WARN_FLAGS})
add_test(NAME "test-test_io_queue" COMMAND test_io_queue)

# Zarr sink integration test
add_my_test(test_zarr_sink test_zarr_sink.c zarr_sink stream CUDA::cudart_static)
target_link_libraries(test_zarr_sink PRIVATE test_platform ${ZSTD_LIBRARY})
target_include_directories(test_zarr_sink PRIVATE ${ZSTD_INCLUDE_DIR})

# Downsample kernel test
add_my_test(test_downsample test_downsample.c downsample CUDA::cudart_static)

# Morton rank test (no CUDA)
add_executable(test_morton morton.c)
target_link_libraries(test_morton PRIVATE lod_plan index_ops chucky_log)
target_compile_options(test_morton PRIVATE ${C_WARN_FLAGS})
add_test(NAME "test-test_morton" COMMAND test_morton)

# LOD GPU test
add_my_test(test_lod test_lod.c lod lod_plan index_ops CUDA::cudart_static)

# Multiscale integration test
add_my_test(test_multiscale test_multiscale.c stream CUDA::cudart_static)
target_link_libraries(test_multiscale PRIVATE test_data zarr_sink ${ZSTD_LIBRARY})
target_include_directories(test_multiscale PRIVATE ${ZSTD_INCLUDE_DIR})

