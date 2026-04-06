// Public zarr v3 array and group interface.
// Create arrays that implement the shard_sink interface for streaming writes.
#pragma once

#include "dimension.h"
#include "dtype.h"
#include "store.h"
#include "types.codec.h"
#include "writer.h"

#include <stdint.h>

struct zarr_array_config
{
  enum dtype data_type;
  double fill_value;
  uint8_t rank;
  const struct dimension* dimensions;
  struct codec_config codec;
};

struct zarr_array;

// Create a zarr v3 array.
// Writes {prefix}/zarr.json. Does NOT write root or intermediate groups.
// The caller must ensure the prefix directory exists (via zarr_write_group
// or the higher-level ngff/hcs layers).
// prefix may be "" to write at the store root.
// Returns NULL on error.
struct zarr_array*
zarr_array_create(struct store* store,
                  const char* prefix,
                  const struct zarr_array_config* cfg);

void
zarr_array_destroy(struct zarr_array* a);

struct shard_sink*
zarr_array_as_shard_sink(struct zarr_array* a);

int
zarr_array_flush(struct zarr_array* a);

int
zarr_array_has_error(const struct zarr_array* a);

// Returns number of bytes queued but not yet written.
size_t
zarr_array_pending_bytes(const struct zarr_array* a);

// Access live dimensions (reflects append-dimension updates).
const struct dimension*
zarr_array_dimensions(const struct zarr_array* a);

// Write a zarr v3 group zarr.json at the given key.
// attributes_json: raw JSON string for the "attributes" field.
//   If NULL, writes an empty attributes object.
// Returns 0 on success, non-zero on error.
int
zarr_write_group(struct store* store,
                 const char* key,
                 const char* attributes_json);
