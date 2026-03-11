# add_test_exe(NAME source.c [LIBS ...])
#   Creates a test executable with warnings, nix rpath, and registers it with ctest.
function(add_test_exe TGT SOURCE_FILE)
    add_executable(${TGT} ${SOURCE_FILE})
    target_link_libraries(${TGT} PRIVATE warnings ${ARGN})
    set_nix_rpath(${TGT})
    add_test(NAME "test-${TGT}" COMMAND ${TGT})
endfunction()

# add_test_lib(TARGET sources...)
#   Creates a STATIC library, exposes its headers, and links warnings.
function(add_test_lib TARGET)
    add_library(${TARGET} STATIC ${ARGN})
    target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${TARGET} PRIVATE warnings)
endfunction()

if(MSVC)
    set(AVX2_FLAG /arch:AVX2)
else()
    set(AVX2_FLAG -mavx2)
endif()

add_test_exe(test_ctx_null test_ctx_null.c CUDA::cuda_driver)

# Index operations utility library (shared between scalar and AVX2 tests)
add_test_lib(index_ops_util index.ops.util.c)
target_link_libraries(index_ops_util PUBLIC index_ops)

add_test_exe(index.ops index.ops.c CUDA::cuda_driver index_ops_util stream)
enable_openmp(index.ops)

# AVX2 helper library
add_test_lib(div_avx2 div.avx2.c)
target_compile_options(div_avx2 PRIVATE ${AVX2_FLAG})

add_test_exe(index.ops.avx2 index.ops.avx2.c CUDA::cuda_driver div_avx2 index_ops_util stream)
target_compile_options(index.ops.avx2 PRIVATE ${AVX2_FLAG})
enable_openmp(index.ops.avx2)

# Magic division test
add_test_exe(test_magic_div test_magic_div.c div_avx2)

# AVX2 division test
add_test_exe(test_avx2_div test_avx2_div.c div_avx2)
target_compile_options(test_avx2_div PRIVATE ${AVX2_FLAG})

# Transpose indices CUDA test
add_test_exe(index.ops.cuda index.ops.cuda.c CUDA::cuda_driver CUDA::cudart_static transpose index_ops_util chucky_log)

# Test platform utilities
add_test_lib(test_platform test_platform.h)
add_platform_sources(test_platform)

# Shared test data generation library
add_test_lib(test_data test_data.c test_data.h)
target_link_libraries(test_data PUBLIC stream)

# Tests with zstd
add_test_exe(test_stream          test_stream.c          CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd index_ops_util)
add_test_exe(test_compress        test_compress.c        CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd)
add_test_exe(test_aggregate       test_aggregate.c       CUDA::cuda_driver CUDA::cudart_static aggregate)
add_test_exe(test_shard_contents  test_shard_contents.c  CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd)

add_test_exe(test_json_writer test_json_writer.c json_writer chucky_log)
add_test_exe(test_io_queue    test_io_queue.c    io_queue chucky_log)

add_test_exe(test_zarr_sink   test_zarr_sink.c   CUDA::cuda_driver CUDA::cudart_static zarr_sink stream Zstd::Zstd test_platform)
add_test_exe(test_downsample  test_downsample.c  CUDA::cuda_driver CUDA::cudart_static downsample)

add_test_exe(test_morton      morton.c           lod_plan index_ops chucky_log)

add_test_exe(test_lod         test_lod.c         CUDA::cuda_driver CUDA::cudart_static lod lod_plan index_ops)
add_test_exe(test_multiscale  test_multiscale.c  CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd test_data zarr_sink)
