// OME-NGFF v0.5 HCS (High-Content Screening) metadata generation.
#pragma once

#include <stddef.h>
#include <stdint.h>

struct attr_set;

// Generate plate-level OME attributes JSON into buf.
// This is the "attributes" value for the plate group zarr.json.
// extras: optional custom attributes spliced alongside the ome block.
// Returns JSON length on success, -1 on error.
int
hcs_plate_attributes_json(char* buf,
                          size_t cap,
                          const char* plate_name,
                          int rows,
                          int cols,
                          const char* row_names,
                          int field_count,
                          const int* well_mask,
                          const struct attr_set* extras);

// Generate well-level OME attributes JSON into buf.
// extras: optional custom attributes spliced alongside the ome block.
// Returns JSON length on success, -1 on error.
int
hcs_well_attributes_json(char* buf,
                         size_t cap,
                         int field_count,
                         const struct attr_set* extras);
