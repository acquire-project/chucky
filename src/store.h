// Public store interface.
// Users create a store (filesystem or S3) and pass it to zarr/ngff/hcs layers.
// The store is opaque — internal operations (put, mkdirs, pool creation) are
// handled by the format layers.
#pragma once

#include "dimension.h"
#include "dtype.h"

#include <stddef.h>
#include <stdint.h>

struct store;

// Create a filesystem store rooted at the given directory.
// unbuffered: use O_DIRECT / FILE_FLAG_NO_BUFFERING for shard data writes.
// Returns NULL on error.
struct store*
store_fs_create(const char* root, int unbuffered);

// --- S3 store ---

struct store_s3_config
{
  const char* bucket;
  const char* prefix; // key prefix (e.g. "data/out.zarr"), may be NULL
  const char* region;
  const char* endpoint;
  size_t part_size;
  double throughput_gbps;
  size_t max_retries;
  uint32_t backoff_scale_ms;
  uint32_t max_backoff_secs;
  uint64_t timeout_ns;
};

// Create an S3 store. Owns the s3_client.
// Returns NULL on error.
struct store*
store_s3_create(const struct store_s3_config* cfg);

// Fill zero fields with S3 transport defaults (part_size=8MiB,
// throughput=10Gbps).
void
store_s3_config_set_defaults(struct store_s3_config* cfg);

// Validate that shard sizes won't exceed S3 multipart upload limits.
// Returns 0 on success, non-zero if any shard would need >10000 parts.
int
store_s3_validate_part_count(uint8_t rank,
                             const struct dimension* dimensions,
                             enum dtype data_type,
                             size_t part_size);

// Coarse overwrite-guard. Returns 1 if the store root already contains zarr
// data (zarr.json at root), 0 if empty, -1 on IO error. O(1) — one stat / HEAD.
// Does NOT enumerate shards.
int
store_has_existing_data(struct store* s);

// Destroy a store created by store_fs_create or store_s3_create.
void
store_destroy(struct store* s);
