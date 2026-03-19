#pragma once

#include "dimension.h"
#include "types.codec.h"
#include "writer.h"
#include "zarr_fs_sink.h" // for zarr_dtype

#include <stdint.h>

struct zarr_s3_config
{
  const char* bucket;     // S3 bucket name
  const char* prefix;     // key prefix (like store_path), no leading/trailing /
  const char* array_name; // e.g. "0"
  const char* region;     // NULL = auto (us-east-1)
  const char* endpoint;   // NULL = AWS default, or e.g. "http://localhost:9000"
  enum zarr_dtype data_type;
  double fill_value;
  uint8_t rank;
  const struct dimension* dimensions;
  enum compression_codec codec; // CODEC_ZSTD, CODEC_LZ4, or CODEC_NONE
};

struct zarr_s3_sink;

// Create a zarr v3 store sink backed by S3.
// Writes zarr.json metadata and shard objects to the bucket.
// Returns NULL on error.
struct zarr_s3_sink*
zarr_s3_sink_create(const struct zarr_s3_config* cfg);

void
zarr_s3_sink_destroy(struct zarr_s3_sink* s);

// Block until all in-flight uploads have completed.
void
zarr_s3_sink_flush(struct zarr_s3_sink* s);

// Get the shard_sink interface for use with the chunk stream.
struct shard_sink*
zarr_s3_sink_as_shard_sink(struct zarr_s3_sink* s);

// --- Multiscale (OME-NGFF) ---

struct zarr_s3_multiscale_config
{
  const char* bucket;
  const char* prefix;
  const char* array_name; // group name; NULL = write at prefix
  const char* region;
  const char* endpoint;
  enum zarr_dtype data_type;
  double fill_value;
  uint8_t rank;
  const struct dimension* dimensions; // L0 dimensions
  int nlod;                           // number of levels (0 = auto)
  enum compression_codec codec;
};

struct zarr_s3_multiscale_sink;

struct zarr_s3_multiscale_sink*
zarr_s3_multiscale_sink_create(const struct zarr_s3_multiscale_config* cfg);

void
zarr_s3_multiscale_sink_destroy(struct zarr_s3_multiscale_sink* s);

void
zarr_s3_multiscale_sink_flush(struct zarr_s3_multiscale_sink* s);

struct shard_sink*
zarr_s3_multiscale_sink_as_shard_sink(struct zarr_s3_multiscale_sink* s);
