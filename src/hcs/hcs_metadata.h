// OME-NGFF v0.5 HCS (High-Content Screening) metadata generation.
#pragma once

#include <stddef.h>
#include <stdint.h>

// Generate plate-level OME attributes JSON into buf.
// This is the "attributes" value for the plate group zarr.json.
// rows/cols: number of rows/columns in the plate.
// row_names: single-char per row (e.g. "ABCD"), NULL = A-Z.
// field_count: number of fields of view per well.
// Only populates wells that are marked active in the well_mask.
// well_mask: rows*cols booleans (row-major), NULL = all active.
// Returns JSON length on success, -1 on error.
int
hcs_plate_attributes_json(char* buf,
                          size_t cap,
                          const char* plate_name,
                          int rows,
                          int cols,
                          const char* row_names,
                          int field_count,
                          const int* well_mask);

// Generate well-level OME attributes JSON into buf.
// This is the "attributes" value for a well group zarr.json.
// field_count: number of FOVs in this well.
// Returns JSON length on success, -1 on error.
int
hcs_well_attributes_json(char* buf, size_t cap, int field_count);
