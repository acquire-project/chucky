// Utility for writing zarr v3 group nodes.
#pragma once

#include "zarr/store.h"

// Write a zarr v3 group zarr.json at the given key.
// attributes_json: raw JSON string for the "attributes" field.
//   If NULL, writes an empty attributes object.
// Returns 0 on success, non-zero on error.
int
zarr_write_group(struct store* store,
                 const char* key,
                 const char* attributes_json);
