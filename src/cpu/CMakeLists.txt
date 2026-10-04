add_src_lib(transpose_cpu SOURCES transpose.cpp transpose.h
    LINKS index_ops threadpool
)

add_src_lib(compress_cpu SOURCES compress.c compress.h compress_blosc.c compress_blosc.h
    LINKS codec_types threadpool Lz4::Lz4 Zstd::Zstd Blosc::Blosc chucky_log
)

add_src_lib(aggregate_cpu SOURCES aggregate.c aggregate.h
    LINKS index_ops threadpool chucky_log
)

add_src_lib(lod_cpu SOURCES lod.cpp lod.h
    LINKS lod_plan index_ops stream_config threadpool chucky_log
)

add_src_lib(pipeline_cpu SOURCES pipeline.c pipeline.h
    LINKS compress_cpu aggregate_cpu lod_cpu
          stream_config shard_delivery threadpool chucky_log
)

add_src_lib(stream_cpu SOURCES stream.c stream.body.c stream.body.h ../stream.cpu.h stream.internal.h
    LINKS transpose_cpu pipeline_cpu threadpool
          stream_config dimension shard_delivery platform chucky_log
)
