// Public OME-NGFF v0.5 HCS (High-Content Screening) plate interface.
// Creates plate -> row -> well -> FOV hierarchy with per-level metadata.
// Each FOV is an ngff_multiscale.
#pragma once

#include "ngff.h"

#include <stddef.h>
#include <stdint.h>

struct hcs_plate_config
{
  const char* name;      // plate name (also used as prefix in store)
  int rows;              // number of rows
  int cols;              // number of columns
  int field_count;       // FOVs per well (uniform across all wells)
  const char* row_names; // single-char per row (e.g. "ABCD"), NULL = A-Z
  const int* well_mask;  // rows*cols booleans (row-major), NULL = all active

  // Per-FOV multiscale config (shared across all FOVs)
  struct ngff_multiscale_config fov;
};

struct hcs_plate;

// Create an HCS plate sink.
// Writes the full hierarchy: root group, plate group (with OME plate attrs),
// row groups, well groups (with OME well attrs), and per-FOV multiscale sinks.
// Returns NULL on error.
struct hcs_plate*
hcs_plate_create(struct store* store, const struct hcs_plate_config* cfg);

// Auto-flushes pending metadata best-effort; ignores flush errors. Call
// hcs_plate_flush_metadata before destroy if you need to detect failures.
void
hcs_plate_destroy(struct hcs_plate* p);

// Get the shard_sink for a specific field of view.
// row/col are 0-based indices. fov is 0-based within the well.
// Returns NULL if the well is not active or indices are out of range.
struct shard_sink*
hcs_plate_fov_sink(struct hcs_plate* p, int row, int col, int fov);

// Buffer a custom attribute on the plate group.
int
hcs_plate_set_attribute(struct hcs_plate* p,
                        const char* attr_key,
                        const char* json_value);

// Buffer a custom attribute on a well group.
int
hcs_plate_set_well_attribute(struct hcs_plate* p,
                             int row,
                             int col,
                             const char* attr_key,
                             const char* json_value);

// Buffer a custom attribute on a single FOV's multiscale group.
int
hcs_plate_set_fov_attribute(struct hcs_plate* p,
                            int row,
                            int col,
                            int fov,
                            const char* attr_key,
                            const char* json_value);

// Force rewrite of all dirty plate/well/FOV group zarr.json files.
int
hcs_plate_flush_metadata(struct hcs_plate* p);
