# add_bench(NAME source.c [LIBS ...])
#   Creates a benchmark executable with warnings and nix rpath.
function(add_bench TGT SOURCE_FILE)
    add_executable(${TGT} ${SOURCE_FILE})
    target_link_libraries(${TGT} PRIVATE warnings ${ARGN})
    set_nix_rpath(${TGT})
endfunction()

# bench_util: static library with pipeline benchmark infrastructure
add_library(bench_util STATIC bench_util.c bench_util.h)
target_include_directories(bench_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    bench_util
    PUBLIC
        stream
        stream_cpu
        zarr_fs_sink
        zarr_s3_sink
        test_data
        CUDA::cuda_driver
    PRIVATE warnings
)

# bench_stream: per-scenario, per-mode executables
set(BENCH_STREAM_SOURCES
    bench_stream_256cube_single.c
    bench_stream_256cube_multiscale.c
    bench_stream_256cube_multiscale_dim0.c
    bench_stream_orca2_single.c
    bench_stream_orca2_multiscale.c
    bench_stream_orca2_multiscale_dim0.c
    bench_stream_medfmt_single.c
    bench_stream_smallepoch_single.c
    bench_stream_256cube_two_streams.c
)

foreach(src IN LISTS BENCH_STREAM_SOURCES)
    string(REPLACE ".c" "" target ${src})
    add_bench(${target} ${src} bench_util CUDA::cudart_static Zstd::Zstd)
endforeach()
