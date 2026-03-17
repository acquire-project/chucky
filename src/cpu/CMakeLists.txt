add_src_lib(transpose_cpu SOURCES transpose.c transpose.h
    LINKS index_ops)
target_include_directories(transpose_cpu PUBLIC ${CMAKE_SOURCE_DIR}/src)
enable_openmp(transpose_cpu)

add_src_lib(compress_cpu SOURCES compress.c compress.h
    LINKS Lz4::Lz4 Zstd::Zstd chucky_log)
target_include_directories(compress_cpu PUBLIC ${CMAKE_SOURCE_DIR}/src)
enable_openmp(compress_cpu)

add_src_lib(aggregate_cpu SOURCES aggregate.c aggregate.h
    LINKS index_ops chucky_log)
target_include_directories(aggregate_cpu PUBLIC ${CMAKE_SOURCE_DIR}/src)
enable_openmp(aggregate_cpu)

add_src_lib(lod_cpu SOURCES lod.cpp lod.h
    LINKS lod_plan index_ops stream_config chucky_log)
target_include_directories(lod_cpu PUBLIC ${CMAKE_SOURCE_DIR}/src)
enable_openmp(lod_cpu)
