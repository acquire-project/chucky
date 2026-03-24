#pragma once

#include "types.stream.h"
#include "writer.h"
#include <cuda.h>

struct tile_stream_layout_gpu
{
  uint64_t* d_lifted_shape;  // device copy (allocated once)
  int64_t* d_lifted_strides; // device copy (allocated once)
};

struct tile_stream_memory_info
{
  size_t device_bytes;      // total GPU memory
  size_t host_pinned_bytes; // total pinned host memory

  // Breakdown (device)
  size_t staging_bytes;         // 2 x buffer_capacity_bytes
  size_t chunk_pool_bytes;      // 2 x total_chunks x chunk_stride x bpe
  size_t compressed_pool_bytes; // 2 x total_chunks x max_output_size
  size_t aggregate_bytes;       // sum over levels: device aggregate buffers
  size_t lod_bytes;             // d_linear + d_morton + shape arrays
  size_t codec_bytes;           // nvcomp workspace + pointer arrays

  // Breakdown (host heap, not pinned)
  size_t shard_bytes; // active_shard arrays + index buffers

  // Key parameters used in the estimate
  uint64_t chunks_per_epoch; // L0
  uint64_t total_chunks;     // sum across all LOD levels
  size_t max_output_size;    // compressed chunk bound
  int nlod;                  // number of LOD levels
  uint32_t epochs_per_batch; // K
};

// Estimate GPU memory requirements without allocating.
// Returns 0 on success, non-zero on invalid config.
int
tile_stream_gpu_memory_estimate(const struct tile_stream_configuration* config,
                                struct tile_stream_memory_info* info);

// Find the largest power-of-2 chunk size (starting from target_chunk_bytes)
// that fits within budget_bytes of GPU device memory.
// Modifies config->dimensions in place. Returns 0 on success.
int
tile_stream_gpu_advise_chunk_sizes(struct tile_stream_configuration* config,
                                   size_t target_chunk_bytes,
                                   const uint8_t* ratios,
                                   size_t budget_bytes);

// Allocate and initialize a tile_stream_gpu. Returns pointer on success,
// NULL on failure. Caller must free with tile_stream_gpu_destroy.
// The config->dimensions pointer must remain valid for the lifetime of the
// stream.
struct tile_stream_gpu*
tile_stream_gpu_create(const struct tile_stream_configuration* config,
                       struct shard_sink* sink);

void
tile_stream_gpu_destroy(struct tile_stream_gpu* stream);

// Return accumulated timing metrics.
struct stream_metrics
tile_stream_gpu_get_metrics(const struct tile_stream_gpu* s);

// --- Accessors ---

const struct tile_stream_layout*
tile_stream_gpu_layout(const struct tile_stream_gpu* s);

struct writer*
tile_stream_gpu_writer(struct tile_stream_gpu* s);

uint64_t
tile_stream_gpu_cursor(const struct tile_stream_gpu* s);

struct tile_stream_status
tile_stream_gpu_status(const struct tile_stream_gpu* s);
