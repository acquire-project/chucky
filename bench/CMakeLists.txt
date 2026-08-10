# add_bench(NAME source.c [LIBS ...])
#   Creates a benchmark executable with warnings and nix rpath.
function(add_bench TGT SOURCE_FILE)
    add_executable(${TGT} ${SOURCE_FILE})
    target_link_libraries(${TGT} PRIVATE warnings ${ARGN})
    set_nix_rpath(${TGT})
endfunction()

# bench_util: static library with pipeline benchmark infrastructure
add_library(
    bench_util
    STATIC
    bench_util.c
    bench_util.h
    bench_gpu.h
    sink_discard.c
    sink_discard.h
    sink_throttled.c
    sink_throttled.h
    sink_metering.c
    sink_metering.h
    bench_zarr.c
    bench_zarr.h
    bench_parse.c
    bench_parse.h
    bench_report.c
    bench_report.h
)
target_include_directories(bench_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    bench_util
    PUBLIC
        stream_cpu
        zarr_array
        ngff_multiscale
        zarr_group
        zarr_metadata
        store_fs
        store_s3
        store_api
        lod_plan
        dimension
        format_bytes
        test_data
    PRIVATE warnings
)

if(CHUCKY_ENABLE_GPU)
    target_sources(bench_util PRIVATE bench_gpu.cuda.c)
    target_link_libraries(
        bench_util
        PUBLIC stream CUDA::cuda_driver CUDA::cudart_static
    )
else()
    target_sources(bench_util PRIVATE bench_gpu.none.c)
endif()

# bench_stream: per-scenario, per-mode executables
set(BENCH_STREAM_SOURCES
    bench_stream_256cube_single.c
    bench_stream_256cube_multiscale.c
    bench_stream_256cube_multiscale_dim0.c
    bench_stream_orca2_single.c
    bench_stream_orca2_multiscale.c
    bench_stream_orca2_multiscale_dim0.c
    bench_stream_medfmt_single.c
    bench_stream_medfmt_multiscale.c
    bench_stream_medfmt_multiscale_dim0.c
    bench_stream_smallepoch_single.c
    bench_stream_aqz_single.c
)

# Two streams share one GPU, so there is nothing to run without one.
if(CHUCKY_ENABLE_GPU)
    list(APPEND BENCH_STREAM_SOURCES bench_stream_256cube_two_streams.c)
endif()

foreach(src IN LISTS BENCH_STREAM_SOURCES)
    string(REPLACE ".c" "" target ${src})
    add_bench(${target} ${src} bench_util Zstd::Zstd)
endforeach()
