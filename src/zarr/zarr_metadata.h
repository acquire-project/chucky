#pragma once

#include "dimension.h"
#include "dtype.h"
#include "types.codec.h"

#include <stddef.h>
#include <stdint.h>

// Generate zarr v3 root group JSON into buf.
// Returns length on success, -1 on error.
int
zarr_root_json(char* buf, size_t cap);

// Generate zarr v3 array JSON into buf.
// Returns length on success, -1 on error.
int
zarr_array_json(char* buf,
                size_t cap,
                uint8_t rank,
                const struct dimension* dimensions,
                enum dtype data_type,
                double fill_value,
                const uint64_t* chunks_per_shard,
                enum compression_codec codec);

// Compute shard key/path suffix: "c/0/1/2" for a flat shard index.
// Writes into buf. Returns 0 on success, -1 on error.
int
zarr_shard_key(char* buf,
               size_t cap,
               uint8_t rank,
               const uint64_t* shard_count,
               uint64_t flat);

// Generate OME-NGFF v0.5 multiscale group JSON into buf.
// level_dims[lv] points to the rank-length dimension array for level lv.
// Returns JSON length on success, -1 on error.
int
zarr_multiscale_group_json(char* buf,
                           size_t cap,
                           uint8_t rank,
                           int nlod,
                           const struct dimension* const* level_dims);

// Compute shard geometry from dimensions.
struct zarr_geometry
{
  uint64_t chunk_count[8];
  uint64_t chunks_per_shard[8];
  uint64_t shard_count[8];
  uint64_t shard_inner_count; // prod(shard_count[d] for d > 0)
};

void
zarr_compute_geometry(struct zarr_geometry* g,
                      uint8_t rank,
                      const struct dimension* dimensions);

void
zarr_compute_geometry_from_shape(struct zarr_geometry* g,
                                 uint8_t rank,
                                 const uint64_t* shape,
                                 const struct dimension* dimensions);
