add_src_lib(transpose_cpu SOURCES transpose.c transpose.h
    LINKS index_ops)
target_include_directories(transpose_cpu PUBLIC ${CMAKE_SOURCE_DIR}/src)
enable_openmp(transpose_cpu)

add_src_lib(compress_cpu SOURCES compress.c compress.h
    LINKS Lz4::Lz4 Zstd::Zstd chucky_log)
target_include_directories(compress_cpu PUBLIC ${CMAKE_SOURCE_DIR}/src)
enable_openmp(compress_cpu)
