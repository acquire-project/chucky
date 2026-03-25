add_src_lib(json_writer SOURCES json_writer.c json_writer.h)

add_src_lib(zarr_metadata SOURCES zarr_metadata.c zarr_metadata.h
    LINKS json_writer dimension chucky_log)

add_src_lib(shard_delivery SOURCES shard_delivery.c shard_delivery.h
    LINKS crc32c writer platform chucky_log)

add_src_lib(zarr_fs_sink SOURCES zarr_fs_sink.c ../zarr_fs_sink.h
    LINKS zarr_metadata writer io_queue platform_io lod_plan chucky_log)

add_src_lib(s3_client SOURCES s3_client.c s3_client.h
    LINKS AWS::aws-c-s3 chucky_log)

add_src_lib(zarr_s3_sink SOURCES zarr_s3_sink.c ../zarr_s3_sink.h
    LINKS zarr_metadata s3_client writer lod_plan chucky_log)
