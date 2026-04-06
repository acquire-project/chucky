add_src_lib(hcs_metadata SOURCES hcs_metadata.c hcs_metadata.h
    LINKS json_writer chucky_log
)

add_src_lib(hcs SOURCES hcs.c
    LINKS hcs_metadata ngff_multiscale zarr_group lod_plan chucky_log
)
