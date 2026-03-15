#pragma once

#include "compress.h"
#include "lod.h"
#include "metric.h"
#include "transpose.h"
#include "writer.h"
#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

struct stream_metrics
{
  struct stream_metric memcpy;
  struct stream_metric h2d;
  struct stream_metric lod_gather;
  struct stream_metric lod_reduce;
  struct stream_metric lod_dim0_fold;
  struct stream_metric lod_morton_tile;
  struct stream_metric scatter;
  struct stream_metric compress;
  struct stream_metric aggregate;
  struct stream_metric d2h;
  struct stream_metric sink;
};

struct dimension
{
  uint64_t size; // 0 means unbounded (dim 0 only: stream indefinitely)
  uint64_t tile_size;
  uint64_t tiles_per_shard; // 0 means all tiles along this dimension
                            // (must be > 0 when size == 0)
  const char* name;         // optional label (e.g. "x"), may be NULL
  int downsample;           // include in LOD pyramid
  uint8_t storage_position; // position in storage layout (0=outermost).
                            // dims[0].storage_position must be 0.
                            // Must be a valid permutation of 0..rank-1.
};

struct tile_stream_configuration
{
  // Size of each H2D staging buffer (double-buffered: 4x total allocation)
  size_t buffer_capacity_bytes;
  size_t bytes_per_element;
  uint8_t rank;
  const struct dimension* dimensions;
  struct shard_sink* shard_sink; // downstream shard writer factory, not owned
  enum compression_codec codec;  // compression codec for tiles
  enum lod_reduce_method reduce_method;      // epoch LOD reduction method
  enum lod_reduce_method dim0_reduce_method; // dim0 LOD reduction
  uint8_t epochs_per_batch; // K: 0 = auto (target_batch_tiles), must be pow2
  uint32_t
    target_batch_tiles; // minimum tiles per compress batch (default 1024)
  float metadata_update_interval_s; // seconds between metadata updates
  size_t
    shard_alignment; // 0 = no padding; platform_page_size() for unbuffered IO
};

struct stream_layout
{
  uint8_t lifted_rank;
  uint64_t lifted_shape[MAX_RANK];
  int64_t lifted_strides[MAX_RANK];

  uint64_t* d_lifted_shape;  // device copy (allocated once)
  int64_t* d_lifted_strides; // device copy (allocated once)

  uint64_t tile_elements;   // elements per tile
  uint64_t tile_stride;     // elements between tile starts (>= tile_elements)
  uint64_t tiles_per_epoch; // M = prod of tile_count[i] for i > 0
  uint64_t epoch_elements;  // elements per epoch = M * tile_elements
  size_t tile_pool_bytes;   // tiles_per_epoch * tile_stride * bpe
};

struct tile_stream_memory_info
{
  size_t device_bytes;      // total GPU memory
  size_t host_pinned_bytes; // total pinned host memory

  // Breakdown (device)
  size_t staging_bytes;         // 2 x buffer_capacity_bytes
  size_t tile_pool_bytes;       // 2 x total_tiles x tile_stride x bpe
  size_t compressed_pool_bytes; // 2 x total_tiles x max_output_size
  size_t aggregate_bytes;       // sum over levels: device aggregate buffers
  size_t lod_bytes;             // d_linear + d_morton + shape arrays
  size_t codec_bytes;           // nvcomp workspace + pointer arrays

  // Key parameters used in the estimate
  uint64_t tiles_per_epoch;  // L0
  uint64_t total_tiles;      // sum across all LOD levels
  size_t max_output_size;    // compressed tile bound
  int nlod;                  // number of LOD levels
  uint32_t epochs_per_batch; // K
};

// Estimate GPU memory requirements without allocating.
// Returns 0 on success, non-zero on invalid config.
int
tile_stream_gpu_memory_estimate(const struct tile_stream_configuration* config,
                                struct tile_stream_memory_info* info);

// Allocate and initialize a tile_stream_gpu. Returns pointer on success,
// NULL on failure. Caller must free with tile_stream_gpu_destroy.
struct tile_stream_gpu*
tile_stream_gpu_create(const struct tile_stream_configuration* config);

void
tile_stream_gpu_destroy(struct tile_stream_gpu* stream);

// Return accumulated timing metrics.
struct stream_metrics
tile_stream_gpu_get_metrics(const struct tile_stream_gpu* s);

// --- Accessors ---

const struct stream_layout*
tile_stream_gpu_layout(const struct tile_stream_gpu* s);

struct writer*
tile_stream_gpu_writer(struct tile_stream_gpu* s);

uint64_t
tile_stream_gpu_cursor(const struct tile_stream_gpu* s);

struct tile_stream_status
{
  int nlod;
  int dim0_downsample;
  uint32_t epochs_per_batch;
  size_t max_compressed_size;
  size_t bytes_per_element;
  enum compression_codec codec;
  size_t codec_batch_size;
  uint32_t batch_accumulated;
  int pool_current;
  int flush_pending;
};

struct tile_stream_status
tile_stream_gpu_status(const struct tile_stream_gpu* s);
