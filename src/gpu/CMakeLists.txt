add_src_lib(transpose SOURCES transpose.cu transpose.h LINKS CUDA::cuda_driver)
add_src_lib(compress  SOURCES compress.cu compress.h   LINKS nvcomp::nvcomp CUDA::cudart_static chucky_log)
add_src_lib(aggregate SOURCES aggregate.cu aggregate.h LINKS CUDA::cudart_static CUDA::cuda_driver stream_config chucky_log)
add_src_lib(lod       SOURCES lod.cu lod.h             LINKS CUDA::cudart_static CUDA::cuda_driver chucky_log)

add_src_lib(stream
    SOURCES
        stream.c ../stream.gpu.h stream.init.c
        stream.flush.c stream.flush.h
        stream.ingest.c stream.ingest.h
        stream.lod.c stream.lod.h
        stream.internal.h
        flush.compress_agg.c flush.compress_agg.h
        flush.d2h_deliver.c flush.d2h_deliver.h
        flush.handoff.h flush.helpers.h
    LINKS
        transpose compress aggregate lod stream_config lod_plan index_ops
        dimension crc32c shard_delivery writer platform chucky_log
        CUDA::cuda_driver)
