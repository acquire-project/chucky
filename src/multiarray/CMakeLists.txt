add_src_lib(multiarray_cpu SOURCES stream.c ../multiarray.cpu.h
    LINKS stream_config transpose_cpu pipeline_cpu
          dimension shard_delivery chucky_log)
