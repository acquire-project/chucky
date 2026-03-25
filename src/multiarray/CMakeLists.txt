add_src_lib(multiarray_cpu SOURCES stream.c ../multiarray.cpu.h
    LINKS stream_config transpose_cpu compress_cpu aggregate_cpu
          lod_cpu dimension shard_delivery chucky_log)
