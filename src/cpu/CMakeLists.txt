add_src_lib(transpose_cpu SOURCES transpose.c transpose.h
    LINKS index_ops)
target_include_directories(transpose_cpu PUBLIC ${CMAKE_SOURCE_DIR}/src)
enable_openmp(transpose_cpu)
