add_src_lib(hcs_metadata SOURCES hcs_metadata.c hcs_metadata.h
    LINKS json_writer attr_set strbuf chucky_log
)

add_src_lib(hcs SOURCES hcs.c
    LINKS hcs_metadata attr_set ngff_multiscale zarr_group lod_plan strbuf chucky_log
)
