// S3-backed shard writer pool.
// Uses async multipart upload with sequence-number fencing.
#pragma once

#include "zarr/shard_pool.h"

#include <stdint.h>

struct s3_client;

// Create an S3 shard pool with nslots writer slots.
// Borrows the client (caller retains ownership).
// prefix: prepended to all shard keys, may be NULL or "".
// Returns NULL on error.
struct shard_pool*
shard_pool_s3_create(struct s3_client* client,
                     const char* bucket,
                     const char* prefix,
                     uint64_t nslots);
