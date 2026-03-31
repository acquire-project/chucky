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

// Walk intermediate path segments of array_name, calling fn for each.
// For array_name = "a/b/c", calls fn("a", ctx) then fn("a/b", ctx).
// Returns 0 on success, first non-zero fn return on failure.
int
zarr_for_each_intermediate(const char* array_name,
                           int (*fn)(const char* partial, void* ctx),
                           void* ctx);

// Generate OME-NGFF v0.5 multiscale group JSON into buf.
// level_dims[lv] points to the rank-length dimension array for level lv.
// Returns JSON length on success, -1 on error.
int
zarr_multiscale_group_json(char* buf,
                           size_t cap,
                           uint8_t rank,
                           int nlod,
                           const struct dimension* const* level_dims);
