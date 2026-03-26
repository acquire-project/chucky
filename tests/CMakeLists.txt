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
target_link_libraries(morton_util PUBLIC lod_plan index_ops CUDA::cuda_driver)

# Shared test shard sink (multi-shard-safe)
add_test_lib(test_shard_sink test_shard_sink.c test_shard_sink.h)
target_link_libraries(test_shard_sink PUBLIC writer)

# Shared GPU test helpers (fill, config builder)
add_test_lib(test_gpu_helpers test_gpu_helpers.c test_gpu_helpers.h)
target_link_libraries(test_gpu_helpers PUBLIC stream CUDA::cuda_driver)

# Shared shard verification helpers (index parse, decompress+verify)
add_test_lib(test_shard_verify test_shard_verify.c test_shard_verify.h)
target_link_libraries(test_shard_verify PUBLIC stream Zstd::Zstd)

# Shared voxel coordinate encoding helpers
add_test_lib(test_voxel_encode test_voxel_encode.c test_voxel_encode.h)

# Index operations unit tests (pure CPU)
add_test_exe(test_index_ops test_index_ops.c index_ops chucky_log)

# Standalone transpose kernel test
add_test_exe(test_transpose test_transpose.c CUDA::cuda_driver CUDA::cudart_static stream index_ops_util)

# CPU transpose test (no CUDA)
add_test_exe(test_transpose_cpu test_transpose_cpu.c transpose_cpu index_ops_util chucky_log)

# CPU compress test (no CUDA)
add_test_exe(test_compress_cpu test_compress_cpu.c compress_cpu Lz4::Lz4 Zstd::Zstd chucky_log)

# CPU aggregate test (no CUDA)
add_test_exe(test_aggregate_cpu test_aggregate_cpu.c aggregate_cpu stream_config chucky_log)

# CPU LOD test (no CUDA)
add_test_exe(test_lod_cpu test_lod_cpu.c lod_cpu chucky_log)

# CPU stream end-to-end test (no CUDA)
add_test_exe(test_stream_cpu test_stream_cpu.c stream_cpu writer test_shard_sink chucky_log)

# CPU multi-array stream test (no CUDA)
add_test_exe(test_multiarray_cpu test_multiarray_cpu.c multiarray_cpu stream_cpu writer test_shard_sink chucky_log)

# Cross-validate GPU vs CPU (byte-exact shard comparison with CODEC_NONE)
add_test_exe(test_cross_validate test_cross_validate.c CUDA::cuda_driver CUDA::cudart_static stream stream_cpu writer)

# Test platform utilities
add_test_lib(test_platform test_platform.h)
add_platform_sources(test_platform)

# Shared test data generation library
add_test_lib(test_data test_data.c test_data.h)
target_link_libraries(test_data PUBLIC stream)

# Tests with zstd
add_test_exe(test_stream          test_stream.c          CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd index_ops_util test_shard_sink test_gpu_helpers test_shard_verify test_voxel_encode)
add_test_exe(test_compress        test_compress.c        CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd)
add_test_exe(test_aggregate       test_aggregate.c       CUDA::cuda_driver CUDA::cudart_static aggregate index_ops_util)

add_test_exe(test_json_writer test_json_writer.c json_writer chucky_log)
add_test_exe(test_io_queue    test_io_queue.c    io_queue chucky_log)

add_test_exe(test_zarr_fs_sink   test_zarr_fs_sink.c   CUDA::cuda_driver CUDA::cudart_static zarr_fs_sink stream Zstd::Zstd test_platform test_shard_verify test_voxel_encode)
add_test_exe(test_zarr_s3_sink   test_zarr_s3_sink.c   zarr_s3_sink platform_cmd chucky_log)

add_test_exe(test_ingest       test_ingest.c       CUDA::cuda_driver CUDA::cudart_static stream index_ops_util)
add_test_exe(test_compress_agg test_compress_agg.c CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd test_gpu_helpers index_ops_util test_shard_verify)
add_test_exe(test_batch        test_batch.c        CUDA::cuda_driver CUDA::cudart_static stream test_shard_sink test_gpu_helpers)
add_test_exe(test_d2h_deliver  test_d2h_deliver.c  CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd test_shard_sink test_gpu_helpers index_ops_util)
add_test_exe(test_flush_orch  test_flush_orchestration.c CUDA::cuda_driver CUDA::cudart_static stream test_shard_sink test_gpu_helpers)
add_test_exe(test_dimension   test_dimension.c   stream chucky_log)
add_test_exe(test_lod         test_lod.c         CUDA::cuda_driver CUDA::cudart_static lod lod_plan index_ops morton_util)
add_test_exe(test_lod_dtypes  test_lod_dtypes.cu CUDA::cuda_driver CUDA::cudart_static lod lod_plan)
add_test_exe(test_multiscale  test_multiscale.c  CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd test_data test_shard_sink)

add_subdirectory(experiments)
