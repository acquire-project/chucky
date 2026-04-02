#pragma once

#include "dimension.h"
#include "dtype.h"
#include "types.codec.h"
#include "writer.h"

#include <stdint.h>

struct zarr_config
{
  const char* store_path; // root directory
  const char* array_name; // e.g. "0"
  enum dtype data_type;
  double fill_value;
  uint8_t rank;
  const struct dimension* dimensions;
  int unbuffered;               // use FILE_FLAG_NO_BUFFERING or O_DIRECT
  enum compression_codec codec; // CODEC_ZSTD, CODEC_LZ4_RAW, or CODEC_NONE
};

struct zarr_fs_sink;

// Create a zarr v3 store sink. Writes directory structure and metadata files.
// Returns NULL on error.
struct zarr_fs_sink*
zarr_fs_sink_create(const struct zarr_config* cfg);

void
zarr_fs_sink_destroy(struct zarr_fs_sink* s);

// Block until all queued I/O has completed.
void
zarr_fs_sink_flush(struct zarr_fs_sink* s);

// Return number of bytes queued but not yet written to disk.
size_t
zarr_fs_sink_pending_bytes(struct zarr_fs_sink* s);

// Get the shard_sink interface for use with the chunk stream.
struct shard_sink*
zarr_fs_sink_as_shard_sink(struct zarr_fs_sink* s);

// --- Multiscale (OME-NGFF) ---

struct zarr_multiscale_config
{
  const char* store_path; // root directory
  const char*
    array_name; // group name (e.g. "multiscale"); NULL → write at store_path
  enum dtype data_type;
  double fill_value;
  uint8_t rank;
  const struct dimension* dimensions; // L0 dimensions
  int nlod;                           // number of levels (0 = auto)
  int unbuffered;                     // use unbuffered IO for shard data
  enum compression_codec codec;       // CODEC_ZSTD, CODEC_LZ4_RAW, or CODEC_NONE
};

struct zarr_fs_multiscale_sink;

// Create a zarr v3 multiscale store. Creates one array per level ("0", "1",
// ...) with correct downsampled shapes, plus OME-NGFF multiscales group
// metadata. Returns NULL on error.
struct zarr_fs_multiscale_sink*
zarr_fs_multiscale_sink_create(const struct zarr_multiscale_config* cfg);

void
zarr_fs_multiscale_sink_destroy(struct zarr_fs_multiscale_sink* s);

void
zarr_fs_multiscale_sink_flush(struct zarr_fs_multiscale_sink* s);

// Return number of bytes queued but not yet written across all levels.
size_t
zarr_fs_multiscale_sink_pending_bytes(struct zarr_fs_multiscale_sink* s);

// Get the shard_sink interface (dispatches open to correct level).
struct shard_sink*
zarr_fs_multiscale_sink_as_shard_sink(struct zarr_fs_multiscale_sink* s);
