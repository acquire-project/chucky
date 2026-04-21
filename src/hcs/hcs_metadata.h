// OME-NGFF v0.5 HCS (High-Content Screening) metadata generation.
#pragma once

#include "util/strbuf.h"

#include <stddef.h>
#include <stdint.h>

struct attr_set;

// Append plate-level OME attributes JSON to sb.
// This is the "attributes" value for the plate group zarr.json.
// extras: optional custom attributes spliced alongside the ome block.
// Returns 0 on success.
int
hcs_plate_attributes_json(struct strbuf* sb,
                          const char* plate_name,
                          int rows,
                          int cols,
                          const char* row_names,
                          int field_count,
                          const int* well_mask,
                          const struct attr_set* extras);

// Append well-level OME attributes JSON to sb.
// extras: optional custom attributes spliced alongside the ome block.
// Returns 0 on success.
int
hcs_well_attributes_json(struct strbuf* sb,
                         int field_count,
                         const struct attr_set* extras);
