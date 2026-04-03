add_src_lib(transpose_cpu SOURCES transpose.cpp transpose.h
    LINKS index_ops
)
enable_openmp(transpose_cpu)

add_src_lib(compress_cpu SOURCES compress.c compress.h compress_blosc.h
    LINKS codec_types Lz4::Lz4 Zstd::Zstd chucky_log
)
if(HAVE_BLOSC)
    target_sources(compress_cpu PRIVATE compress_blosc.c)
    target_link_libraries(compress_cpu PUBLIC Blosc::Blosc)
    target_compile_definitions(compress_cpu PUBLIC HAVE_BLOSC)
else()
    target_sources(compress_cpu PRIVATE compress_blosc_stub.c)
endif()
enable_openmp(compress_cpu)

add_src_lib(aggregate_cpu SOURCES aggregate.c aggregate.h
    LINKS index_ops chucky_log
)
enable_openmp(aggregate_cpu)

add_src_lib(lod_cpu SOURCES lod.cpp lod.h
    LINKS lod_plan index_ops stream_config chucky_log
)
enable_openmp(lod_cpu)

add_src_lib(pipeline_cpu SOURCES pipeline.c pipeline.h
    LINKS compress_cpu aggregate_cpu lod_cpu
          stream_config shard_delivery chucky_log
)
enable_openmp(pipeline_cpu)

add_src_lib(stream_cpu SOURCES stream.c ../stream.cpu.h stream.internal.h
    LINKS transpose_cpu pipeline_cpu
          stream_config dimension shard_delivery platform chucky_log
)
enable_openmp(stream_cpu)
