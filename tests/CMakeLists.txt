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
add_test_lib(morton_util morton.util.c morton.util.h)
target_link_libraries(morton_util PUBLIC lod_plan index_ops)

# Shared test shard sink (multi-shard-safe)
add_test_lib(test_shard_sink test_shard_sink.c test_shard_sink.h)
target_link_libraries(test_shard_sink PUBLIC writer)

# Test-only counting sink shim (wraps a sink, counts copy vs zero-copy paths)
add_test_lib(test_counting_sink test_counting_sink.c test_counting_sink.h)
target_link_libraries(test_counting_sink PUBLIC writer)

# Shared GPU test helpers (fill, config builder)
add_gpu_test_lib(test_gpu_helpers test_gpu_helpers.c test_gpu_helpers.h LINKS stream CUDA::cuda_driver)

# Shared shard verification helpers (index parse, decompress+verify)
add_test_lib(test_shard_verify test_shard_verify.c test_shard_verify.h)
target_link_libraries(test_shard_verify PUBLIC chucky_log Zstd::Zstd)

# Shared voxel coordinate encoding helpers
add_test_lib(test_voxel_encode test_voxel_encode.c test_voxel_encode.h)

# Aggregate-result contract verifier (used by CPU and GPU contract tests)
add_test_lib(aggregate_contract aggregate_contract.c aggregate_contract.h)
target_link_libraries(aggregate_contract PUBLIC stream_config chucky_log)

# Shared zarr sink test helpers (store + pool + array/multiscale setup)
add_test_lib(test_zarr_helpers test_zarr_helpers.c test_zarr_helpers.h)
target_link_libraries(
    test_zarr_helpers
    PUBLIC
        zarr_array
        ngff_multiscale
        zarr_group
        zarr_metadata
        store_fs
        store_api
        lod_plan
        dimension
        chucky_log
)

# Threadpool unit tests
add_test_exe(test_threadpool test_threadpool.c threadpool platform)

# Index operations unit tests (pure CPU)
add_test_exe(test_index_ops test_index_ops.c index_ops chucky_log)

# Standalone transpose kernel test
add_gpu_test_exe(test_transpose test_transpose.c LINKS CUDA::cuda_driver CUDA::cudart_static stream index_ops_util)

# CPU transpose test (no CUDA)
add_test_exe(test_transpose_cpu test_transpose_cpu.c transpose_cpu threadpool index_ops_util chucky_log)

# CPU compress test (no CUDA)
add_test_exe(test_compress_cpu test_compress_cpu.c compress_cpu threadpool chucky_log)

# CPU aggregate test (no CUDA)
add_test_exe(test_aggregate_cpu test_aggregate_cpu.c aggregate_cpu threadpool stream_config chucky_log)

# Aggregate-result contract test (CPU)
add_test_exe(test_aggregate_contract_cpu test_aggregate_contract_cpu.c
    aggregate_cpu shard_delivery aggregate_contract threadpool stream_config lod_plan chucky_log
)

# Aggregate-result contract test (GPU)
add_gpu_test_exe(test_aggregate_contract_gpu test_aggregate_contract_gpu.c
    LINKS CUDA::cuda_driver CUDA::cudart_static aggregate aggregate_contract stream_config lod_plan chucky_log
)

# CPU LOD test (no CUDA)
add_test_exe(test_lod_cpu test_lod_cpu.c lod_cpu compress_cpu morton_util threadpool platform chucky_log)

# CPU stream end-to-end test (no CUDA)
add_test_exe(test_stream_cpu test_stream_cpu.c stream_cpu writer test_shard_sink chucky_log)

# Regression: writer_flush propagates sink->flush errors.
add_test_exe(test_sink_flush_error test_sink_flush_error.c stream_cpu writer test_shard_sink chucky_log)

# Regression: writer_flush cascades through ngff_multiscale to child arrays.
add_test_exe(test_ngff_flush_cascade test_ngff_flush_cascade.c stream_cpu writer ngff_multiscale zarr_array zarr_group store_fs store_api test_platform chucky_log)

# CPU multi-array stream test (no CUDA)
add_test_exe(test_multiarray_cpu test_multiarray_cpu.c multiarray_cpu stream_cpu writer test_shard_sink chucky_log)

# GPU multi-array stream test
add_gpu_test_exe(test_multiarray_gpu test_multiarray_gpu.c LINKS CUDA::cuda_driver CUDA::cudart_static multiarray_gpu stream writer test_shard_sink)

# CPU stream with n_append > 1 (no CUDA)
add_test_exe(test_multi_append test_multi_append.c stream_cpu writer test_shard_sink chucky_log)

# CPU stream + zarr store integration (exercises async IO fencing)
add_test_exe(test_cpu_zarr_fs test_cpu_zarr_fs.c stream_cpu test_zarr_helpers test_counting_sink writer Zstd::Zstd test_platform test_shard_verify)

# Zarr readback validation via zarr-python (requires uv)
add_test_exe(test_zarr_readback test_zarr_readback.c stream_cpu test_zarr_helpers writer test_platform chucky_log)
target_compile_definitions(
    test_zarr_readback
    PRIVATE SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
set_tests_properties(test-test_zarr_readback PROPERTIES SKIP_RETURN_CODE 77)

# Cross-validate GPU vs CPU (byte-exact shard comparison with CODEC_NONE)
add_gpu_test_exe(test_cross_validate test_cross_validate.c LINKS CUDA::cuda_driver CUDA::cudart_static stream stream_cpu writer test_data test_shard_verify)

# Test platform utilities
add_test_lib(test_platform test_platform.h)
add_platform_sources(test_platform)
target_link_libraries(test_platform PUBLIC Threads::Threads)

# Shared test data generation library
add_test_lib(test_data test_data.c test_data.h)
target_link_libraries(test_data PUBLIC stream_config writer chucky_log)

# Tests with zstd
add_gpu_test_exe(test_stream          test_stream.c          LINKS CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd index_ops_util test_shard_sink test_gpu_helpers test_shard_verify test_voxel_encode)
add_gpu_test_exe(test_compress        test_compress.c        LINKS CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd)

add_test_exe(test_json_writer test_json_writer.c zarr_metadata ngff_metadata strbuf chucky_log)
add_test_exe(test_io_queue    test_io_queue.c    io_queue chucky_log)
add_test_exe(test_store_fs    test_store_fs.c    store_fs store_api zarr_group test_platform chucky_log)
add_test_exe(test_zarr_attribute test_zarr_attribute.c zarr_array zarr_group store_fs store_api stream_cpu writer test_platform chucky_log)
add_test_exe(test_ngff_attribute test_ngff_attribute.c ngff_multiscale store_fs store_api test_platform chucky_log)
add_test_exe(test_hcs_attribute test_hcs_attribute.c hcs store_fs store_api test_platform chucky_log)
add_test_exe(test_zarr_array  test_zarr_array.c  zarr_array zarr_group store_fs store_api test_platform chucky_log)
add_test_exe(test_ngff_multiscale test_ngff_multiscale.c ngff_multiscale zarr_group store_fs store_api test_platform chucky_log)
add_test_exe(test_hcs test_hcs.c hcs store_fs store_api test_platform chucky_log)

# OME-NGFF validation via ome-zarr-models (requires uv)
add_test_exe(test_ome_validate test_ome_validate.c hcs ngff_multiscale zarr_group store_fs store_api test_platform chucky_log)
target_compile_definitions(
    test_ome_validate
    PRIVATE SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

add_gpu_test_exe(test_zarr_fs_sink   test_zarr_fs_sink.c   LINKS CUDA::cuda_driver CUDA::cudart_static test_zarr_helpers stream Zstd::Zstd test_platform test_shard_verify test_voxel_encode)
if(CHUCKY_ENABLE_GPU)
    set_tests_properties(test-test_zarr_fs_sink PROPERTIES TIMEOUT 120)
endif()
add_gpu_test_exe(test_destroy_drain  test_destroy_drain.c  LINKS CUDA::cuda_driver CUDA::cudart_static stream zarr_array store_fs store_api shard_pool_fs platform test_platform Zstd::Zstd chucky_log)
add_gpu_test_exe(test_flush_drain_gpu test_flush_drain_gpu.c LINKS CUDA::cuda_driver CUDA::cudart_static stream zarr_array store_fs store_api shard_pool_fs platform test_platform Zstd::Zstd chucky_log)
add_gpu_test_exe(test_destroy_midstream test_destroy_midstream.c LINKS CUDA::cuda_driver CUDA::cudart_static stream zarr_array store_fs store_api shard_pool_fs platform test_platform Zstd::Zstd chucky_log)
add_test_exe(test_flush_drain_cpu test_flush_drain_cpu.c stream_cpu zarr_array store_fs store_api shard_pool_fs platform writer test_platform chucky_log)
add_test_exe(test_flush_error_drain_cpu test_flush_error_drain_cpu.c stream_cpu zarr_array store_fs store_api shard_pool_fs platform writer test_platform chucky_log)
add_gpu_test_exe(test_multiarray_flush_mock  test_multiarray_flush_mock.c  LINKS CUDA::cuda_driver CUDA::cudart_static multiarray_gpu stream writer chucky_log)
add_test_exe(test_zarr_s3_sink   test_zarr_s3_sink.c   zarr_array ngff_multiscale zarr_group zarr_metadata store_s3 store_api lod_plan dimension platform_cmd chucky_log)
add_test_exe(test_store_s3       test_store_s3.c       store_s3 store_api platform_cmd chucky_log)

add_gpu_test_exe(test_ingest       test_ingest.c       LINKS CUDA::cuda_driver CUDA::cudart_static stream index_ops_util)
add_gpu_test_exe(test_compress_agg test_compress_agg.c LINKS CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd test_gpu_helpers index_ops_util test_shard_verify)
add_gpu_test_exe(test_batch        test_batch.c        LINKS CUDA::cuda_driver CUDA::cudart_static stream test_shard_sink test_gpu_helpers)
add_gpu_test_exe(test_d2h_deliver  test_d2h_deliver.c  LINKS CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd test_shard_sink test_gpu_helpers index_ops_util)
add_gpu_test_exe(test_flush_orch  test_flush_orchestration.c LINKS CUDA::cuda_driver CUDA::cudart_static stream test_shard_sink test_gpu_helpers)
add_test_exe(test_dimension   test_dimension.c   stream_config dimension chucky_log)
add_test_exe(test_lz4_warning test_lz4_warning.c stream_config dimension platform chucky_log)
add_test_exe(test_chucky_log  test_chucky_log.c  chucky_log)
add_test_exe(test_strbuf      test_strbuf.c      strbuf chucky_log)
add_test_exe(test_append_latency test_append_latency.c stream_config chucky_log)
add_gpu_test_exe(test_lod         test_lod.c         LINKS CUDA::cuda_driver CUDA::cudart_static lod lod_plan index_ops morton_util)
add_gpu_test_exe(test_lod_dtypes  test_lod_dtypes.cu LINKS CUDA::cuda_driver CUDA::cudart_static lod lod_plan)
add_gpu_test_exe(test_multiscale  test_multiscale.c  LINKS CUDA::cuda_driver CUDA::cudart_static stream Zstd::Zstd test_data test_shard_sink)

add_subdirectory(experiments)
