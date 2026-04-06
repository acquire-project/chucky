// S3-backed store implementation.
#pragma once

#include "dimension.h"
#include "dtype.h"
#include "zarr/store.h"

#include <stddef.h>
#include <stdint.h>

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
