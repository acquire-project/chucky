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

# GPU-only wrappers: no-op when CHUCKY_ENABLE_GPU is OFF.
# Both accept: TARGET sources... [LINKS libs...]
function(add_gpu_test_exe TGT)
    if(CHUCKY_ENABLE_GPU)
        cmake_parse_arguments(ARG "" "" "LINKS" ${ARGN})
        add_test_exe(${TGT} ${ARG_UNPARSED_ARGUMENTS} ${ARG_LINKS})
    endif()
endfunction()

function(add_gpu_test_lib TARGET)
    if(CHUCKY_ENABLE_GPU)
        cmake_parse_arguments(ARG "" "" "LINKS" ${ARGN})
        add_test_lib(${TARGET} ${ARG_UNPARSED_ARGUMENTS})
        if(ARG_LINKS)
            target_link_libraries(${TARGET} PUBLIC ${ARG_LINKS})
        endif()
    endif()
endfunction()

# Index operations utility library (shared between tests)
add_test_lib(index_ops_util index.ops.util.c)
target_link_libraries(index_ops_util PUBLIC index_ops)

# Morton/LOD CPU reference library (shared between tests)
add_gpu_test_lib(morton_util morton.util.c morton.util.h LINKS lod_plan index_ops CUDA::cuda_driver)

# Shared test shard sink (multi-shard-safe)
add_test_lib(test_shard_sink test_shard_sink.c test_shard_sink.h)
target_link_libraries(test_shard_sink PUBLIC writer)

# Shared GPU test helpers (fill, config builder)
add_gpu_test_lib(test_gpu_helpers test_gpu_helpers.c test_gpu_helpers.h LINKS stream CUDA::cuda_driver)

# Shared shard verification helpers (index parse, decompress+verify)
add_test_lib(test_shard_verify test_shard_verify.c test_shard_verify.h)
target_link_libraries(test_shard_verify PUBLIC chucky_log Zstd::Zstd)

# Shared voxel coordinate encoding helpers
add_test_lib(test_voxel_encode test_voxel_encode.c test_voxel_encode.h)

# Index operations unit tests (pure CPU)
add_test_exe(test_index_ops test_index_ops.c index_ops chucky_log)

# Standalone transpose kernel test
add_gpu_test_exe(test_transpose test_transpose.c LINKS CUDA::cuda_driver CUDA::cudart_static stream index_ops_util)

# CPU transpose test (no CUDA)
add_test_exe(test_transpose_cpu test_transpose_cpu.c transpose_cpu index_ops_util chucky_log)
enable_openmp(test_transpose_cpu)

# CPU compress test (no CUDA)
add_test_exe(test_compress_cpu test_compress_cpu.c compress_cpu chucky_log)
enable_openmp(test_compress_cpu)

# CPU aggregate test (no CUDA)
add_test_exe(test_aggregate_cpu test_aggregate_cpu.c aggregate_cpu stream_config chucky_log)
enable_openmp(test_aggregate_cpu)

# CPU LOD test (no CUDA)
add_test_exe(test_lod_cpu test_lod_cpu.c lod_cpu chucky_log)
enable_openmp(test_lod_cpu)

# CPU stream end-to-end test (no CUDA)
add_test_exe(test_stream_cpu test_stream_cpu.c stream_cpu writer test_shard_sink chucky_log)

# CPU multi-array stream test (no CUDA)
add_test_exe(test_multiarray_cpu test_multiarray_cpu.c multiarray_cpu stream_cpu writer test_shard_sink chucky_log)

# CPU stream with n_append > 1 (no CUDA)
add_test_exe(test_multi_append test_multi_append.c stream_cpu writer test_shard_sink chucky_log)

# CPU stream + zarr_fs_sink integration (exercises async IO fencing)
add_test_exe(test_cpu_zarr_fs test_cpu_zarr_fs.c stream_cpu zarr_fs_sink writer Zstd::Zstd test_platform test_shard_verify)

# Zarr readback validation via zarr-python (requires uv)
add_test_exe(test_zarr_readback test_zarr_readback.c stream_cpu zarr_fs_sink writer dimension test_platform chucky_log)
target_compile_definitions(
    test_zarr_readback
    PRIVATE SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
set_tests_properties(test-test_zarr_readback PROPERTIES SKIP_RETURN_CODE 77)

# Cross-validate GPU vs CPU (byte-exact shard comparison with CODEC_NONE)
add_gpu_test_exe(test_cross_validate test_cross_validate.c LINKS CUDA::cuda_driver CUDA::cudart_static stream stream_cpu writer)

# Test platform utilities
add_test_lib(test_platform test_platform.h)
add_platform_sources(test_platform)

# Shared test data generation library
add_test_lib(test_data test_data.c test_data.h)
target_link_libraries(test_data PUBLIC stream_config writer chucky_log)

# Tests with zstd
add_gpu_test_exe(test_stream          test_stream.c          LINKS CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd index_ops_util test_shard_sink test_gpu_helpers test_shard_verify test_voxel_encode)
add_gpu_test_exe(test_compress        test_compress.c        LINKS CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd)
add_gpu_test_exe(test_aggregate       test_aggregate.c       LINKS CUDA::cuda_driver CUDA::cudart_static aggregate index_ops_util)

add_test_exe(test_json_writer test_json_writer.c json_writer chucky_log)
add_test_exe(test_io_queue    test_io_queue.c    io_queue chucky_log)

add_gpu_test_exe(test_zarr_fs_sink   test_zarr_fs_sink.c   LINKS CUDA::cuda_driver CUDA::cudart_static zarr_fs_sink stream Zstd::Zstd test_platform test_shard_verify test_voxel_encode)
add_test_exe(test_zarr_s3_sink   test_zarr_s3_sink.c   zarr_s3_sink platform_cmd chucky_log)

add_gpu_test_exe(test_ingest       test_ingest.c       LINKS CUDA::cuda_driver CUDA::cudart_static stream index_ops_util)
add_gpu_test_exe(test_compress_agg test_compress_agg.c LINKS CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd test_gpu_helpers index_ops_util test_shard_verify)
add_gpu_test_exe(test_batch        test_batch.c        LINKS CUDA::cuda_driver CUDA::cudart_static stream test_shard_sink test_gpu_helpers)
add_gpu_test_exe(test_d2h_deliver  test_d2h_deliver.c  LINKS CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd test_shard_sink test_gpu_helpers index_ops_util)
add_gpu_test_exe(test_flush_orch  test_flush_orchestration.c LINKS CUDA::cuda_driver CUDA::cudart_static stream test_shard_sink test_gpu_helpers)
add_test_exe(test_dimension   test_dimension.c   stream_config dimension chucky_log)
add_gpu_test_exe(test_lod         test_lod.c         LINKS CUDA::cuda_driver CUDA::cudart_static lod lod_plan index_ops morton_util)
add_gpu_test_exe(test_lod_dtypes  test_lod_dtypes.cu LINKS CUDA::cuda_driver CUDA::cudart_static lod lod_plan)
add_gpu_test_exe(test_multiscale  test_multiscale.c  LINKS CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd test_data test_shard_sink)

add_subdirectory(experiments)
