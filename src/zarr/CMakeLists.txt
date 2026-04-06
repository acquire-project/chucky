add_src_lib(json_writer SOURCES json_writer.c json_writer.h)

add_src_lib(zarr_metadata SOURCES zarr_metadata.c zarr_metadata.h
    LINKS json_writer dimension dtype codec_types chucky_log
)

add_src_lib(shard_delivery SOURCES shard_delivery.c shard_delivery.h
    LINKS crc32c writer platform chucky_log
)

add_src_lib(shard_pool_fs SOURCES shard_pool_fs.c shard_pool_fs.h shard_pool.h
    LINKS io_queue platform platform_io writer chucky_log
)

add_src_lib(store_fs SOURCES store_fs.c store_fs.h store.h
    LINKS shard_pool_fs platform_io chucky_log
)

add_src_lib(zarr_group SOURCES zarr_group.c zarr_group.h
    LINKS zarr_metadata chucky_log
)

add_src_lib(zarr_array SOURCES zarr_array.c zarr_array.h
    LINKS zarr_metadata lod_plan writer chucky_log
)

add_src_lib(s3_client SOURCES s3_client.c s3_client.h
    LINKS AWS::aws-c-s3 chucky_log
)

add_src_lib(shard_pool_s3 SOURCES shard_pool_s3.c shard_pool_s3.h
    LINKS s3_client writer chucky_log
)

add_src_lib(store_s3 SOURCES store_s3.c store_s3.h
    LINKS shard_pool_s3 s3_client dtype chucky_log
)
