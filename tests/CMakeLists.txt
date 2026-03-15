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

# Index operations utility library (shared between tests)
add_test_lib(index_ops_util index.ops.util.c)
target_link_libraries(index_ops_util PUBLIC index_ops)

# Morton/LOD CPU reference library (shared between tests)
add_test_lib(morton_util morton.util.c morton.util.h)
target_link_libraries(morton_util PUBLIC lod_plan index_ops)

# Shared test shard sink (multi-shard-safe)
add_test_lib(test_shard_sink test_shard_sink.c test_shard_sink.h)
target_link_libraries(test_shard_sink PUBLIC stream)

# Shared GPU test helpers (fill, config builder)
add_test_lib(test_gpu_helpers test_gpu_helpers.c test_gpu_helpers.h)
target_link_libraries(test_gpu_helpers PUBLIC stream CUDA::cuda_driver)

# Index operations unit tests (pure CPU)
add_test_exe(test_index_ops test_index_ops.c index_ops chucky_log)

# Standalone transpose kernel test
add_test_exe(test_transpose test_transpose.c CUDA::cuda_driver CUDA::cudart_static stream index_ops_util)

# Test platform utilities
add_test_lib(test_platform test_platform.h)
add_platform_sources(test_platform)

# Shared test data generation library
add_test_lib(test_data test_data.c test_data.h)
target_link_libraries(test_data PUBLIC stream)

# Tests with zstd
add_test_exe(test_stream          test_stream.c          CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd index_ops_util test_shard_sink test_gpu_helpers)
add_test_exe(test_compress        test_compress.c        CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd)
add_test_exe(test_aggregate       test_aggregate.c       CUDA::cuda_driver CUDA::cudart_static aggregate index_ops_util)
add_test_exe(test_shard_contents  test_shard_contents.c  CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd)

add_test_exe(test_json_writer test_json_writer.c json_writer chucky_log)
add_test_exe(test_io_queue    test_io_queue.c    io_queue chucky_log)

add_test_exe(test_zarr_sink   test_zarr_sink.c   CUDA::cuda_driver CUDA::cudart_static zarr_sink stream Zstd::Zstd test_platform)

add_test_exe(test_ingest       test_ingest.c       CUDA::cuda_driver CUDA::cudart_static stream index_ops_util)
add_test_exe(test_compress_agg test_compress_agg.c CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd test_gpu_helpers index_ops_util)
add_test_exe(test_batch        test_batch.c        CUDA::cuda_driver CUDA::cudart_static stream test_shard_sink test_gpu_helpers)
add_test_exe(test_d2h_deliver  test_d2h_deliver.c  CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd test_shard_sink test_gpu_helpers index_ops_util)
add_test_exe(test_flush_orch  test_flush_orchestration.c CUDA::cuda_driver CUDA::cudart_static stream test_shard_sink test_gpu_helpers)
add_test_exe(test_dimension   test_dimension.c   stream chucky_log)
add_test_exe(test_lod         test_lod.c         CUDA::cuda_driver CUDA::cudart_static lod lod_plan index_ops morton_util)
add_test_exe(test_multiscale  test_multiscale.c  CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd test_data zarr_sink test_shard_sink)

add_subdirectory(experiments)
