#pragma once

#include "dimension.h"
#include "dtype.h"
#include "types.codec.h"
#include "util/strbuf.h"

#include <stddef.h>
#include <stdint.h>

struct attr_set;

// Append zarr v3 root group JSON to sb. Returns 0 on success.
int
zarr_root_json(struct strbuf* sb);

// Append zarr v3 array JSON to sb.
// extras: optional pre-validated custom attributes to splice into the
// "attributes" object. NULL means emit an empty object.
// Returns 0 on success.
int
zarr_array_json(struct strbuf* sb,
                uint8_t rank,
                const struct dimension* dimensions,
                enum dtype data_type,
                double fill_value,
                const uint64_t* chunks_per_shard,
                struct codec_config codec,
                const struct attr_set* extras);

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
