#pragma once

#include "stream.h"

#include <stdint.h>

enum zarr_dtype
{
  zarr_dtype_uint8,
  zarr_dtype_uint16,
  zarr_dtype_uint32,
  zarr_dtype_uint64,
  zarr_dtype_int8,
  zarr_dtype_int16,
  zarr_dtype_int32,
  zarr_dtype_int64,
  zarr_dtype_float32,
  zarr_dtype_float64,
};

struct zarr_config
{
  const char* store_path; // root directory
  const char* array_name; // e.g. "0"
  enum zarr_dtype data_type;
  double fill_value;
  uint8_t rank;
  const struct dimension* dimensions;
  int unbuffered; // use FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH
};

struct zarr_sink;

// Create a zarr v3 store sink. Writes directory structure and metadata files.
// Returns NULL on error.
struct zarr_sink* zarr_sink_create(const struct zarr_config* cfg);

void zarr_sink_destroy(struct zarr_sink* s);

// Block until all queued I/O has completed.
void zarr_sink_flush(struct zarr_sink* s);

// Return number of bytes queued but not yet written to disk.
size_t zarr_sink_pending_bytes(struct zarr_sink* s);

// Get the shard_sink interface for use with tile_stream_gpu.
struct shard_sink* zarr_sink_as_shard_sink(struct zarr_sink* s);

// --- Multiscale (OME-NGFF) ---

struct zarr_multiscale_config
{
  const char* store_path;  // root directory
  const char* array_name;  // group name (e.g. "multiscale"); NULL → write at store_path
  enum zarr_dtype data_type;
  double fill_value;
  uint8_t rank;
  const struct dimension* dimensions; // L0 dimensions
  int nlod;                           // number of levels (0 = auto)
  int exclude_dim0;                   // exclude dim 0 from LOD (temporal)
  int unbuffered;                     // use unbuffered IO for shard data
};

struct zarr_multiscale_sink;

// Create a zarr v3 multiscale store. Creates one array per level ("0", "1", ...)
// with correct downsampled shapes, plus OME-NGFF multiscales group metadata.
// Returns NULL on error.
struct zarr_multiscale_sink*
zarr_multiscale_sink_create(const struct zarr_multiscale_config* cfg);

void zarr_multiscale_sink_destroy(struct zarr_multiscale_sink* s);

void zarr_multiscale_sink_flush(struct zarr_multiscale_sink* s);

// Return number of bytes queued but not yet written across all levels.
size_t zarr_multiscale_sink_pending_bytes(struct zarr_multiscale_sink* s);

// Get the shard_sink interface (dispatches open to correct level).
struct shard_sink*
zarr_multiscale_sink_as_shard_sink(struct zarr_multiscale_sink* s);
