add_src_lib(multiarray_cpu SOURCES stream.c ../multiarray.cpu.h multiarray.h
    LINKS stream_config transpose_cpu pipeline_cpu threadpool
          dimension shard_delivery chucky_log
)

if(CHUCKY_ENABLE_GPU)
    add_src_lib(multiarray_gpu SOURCES stream.gpu.c ../multiarray.gpu.h multiarray.h
        LINKS stream chucky_log
    )
endif()
