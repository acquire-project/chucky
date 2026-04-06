// Storage-agnostic zarr v3 array format layer.
// Implements shard_sink. Uses store for metadata, shard_pool for data I/O.
#pragma once

#include "dimension.h"
#include "dtype.h"
#include "types.codec.h"
#include "writer.h"
#include "zarr/shard_pool.h"
#include "zarr/store.h"

#include <stddef.h>
#include <stdint.h>

struct zarr_array_config
{
  enum dtype data_type;
  double fill_value;
  uint8_t rank;
  const struct dimension* dimensions;
  struct codec_config codec;

  // Pre-computed geometry (caller provides, e.g. from lod_plan).
  const uint64_t* shard_counts;     // rank elements
  const uint64_t* chunks_per_shard; // rank elements
  uint64_t shard_inner_count;
};

struct zarr_array;

// Only writes {prefix}/zarr.json (the array node).
// Does NOT write root groups or intermediate groups.
// The caller must ensure the prefix directory exists (via store->mkdirs).
// prefix may be "" to write at the store root.
// Returns NULL on error.
struct zarr_array*
zarr_array_create(struct store* store,
                  struct shard_pool* pool,
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

// Access live dimensions (for parent layers to read after update_append).
const struct dimension*
zarr_array_dimensions(const struct zarr_array* a);
