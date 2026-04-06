add_src_lib(ngff_metadata SOURCES ngff_metadata.c ngff_metadata.h
    LINKS json_writer dimension chucky_log
)

add_src_lib(ngff_multiscale SOURCES ngff_multiscale.c ngff_multiscale.h
    LINKS ngff_metadata zarr_array zarr_group zarr_metadata lod_plan dimension chucky_log
)
