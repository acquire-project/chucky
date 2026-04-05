// S3-backed store implementation.
#pragma once

#include "zarr/store.h"

#include <stddef.h>
#include <stdint.h>

struct store_s3_config
{
  const char* bucket;
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
