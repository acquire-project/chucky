// Private group helpers used by ngff/hcs layers.
#pragma once

#include "zarr/store.h"

// Internal: write a zarr v3 group zarr.json at the given key with the supplied
// raw attributes JSON spliced in. attributes_json must be a valid JSON object
// text (e.g. produced by hcs_*_attributes_json). Used by HCS plate/well group
// writers. Returns 0 on success, non-zero on error.
int
zarr_group_write_with_raw_attrs(struct store* store,
                                const char* key,
                                const char* attributes_json);
