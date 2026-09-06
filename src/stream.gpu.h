#pragma once

#include "types.stream.h"
#include "writer.h"
#include <cuda.h>

struct tile_stream_layout;

struct tile_stream_memory_info
{
  size_t device_bytes;      // total GPU memory
  size_t host_pinned_bytes; // total pinned host memory

  // Breakdown (device)
  size_t staging_bytes;         // 2 x buffer_capacity_bytes
  size_t chunk_pool_bytes;      // 2 x total_chunks x chunk_stride x bpe
  size_t compressed_pool_bytes; // 2 x codec_batch x device output stride
  size_t aggregate_bytes;       // sum over levels: device aggregate buffers
  size_t lod_bytes;             // d_linear + d_morton + shape arrays
  size_t codec_bytes; // nvCOMP workspace + pointer/size arrays + filter scratch

  // Breakdown (host heap, not pinned)
  size_t shard_bytes; // active_shard arrays + index buffers

  size_t host_output_pool_bytes;

  // Key parameters used in the estimate
  uint64_t chunks_per_epoch; // L0
  uint64_t total_chunks;     // sum across all LOD levels
  size_t max_output_size;    // compressed chunk bound
  size_t host_output_bytes;
  int nlod;                  // number of LOD levels
  uint32_t epochs_per_batch; // K
};

// Estimate GPU memory requirements without allocating.
// shard_alignment: required write alignment for the I/O backend (e.g. page
//   size for O_DIRECT). 0 = no alignment constraint.
// Returns 0 on success, non-zero on invalid config.
int
tile_stream_gpu_memory_estimate(const struct tile_stream_configuration* config,
                                size_t shard_alignment,
                                struct tile_stream_memory_info* info);

// Solve chunk + shard layout for the GPU backend.
//
// Phase 1: starting from target_chunk_bytes, halves chunk bytes until the
//   device memory estimate fits within budget_bytes or target falls below
//   max(min_chunk_bytes, bpe). At each chunk size, if the auto-derived
//   epochs_per_batch (K) overshoots the budget, K is halved (down to 1)
//   before shrinking chunks further. A non-zero config->epochs_per_batch
//   on entry is treated as user-authoritative and is not reduced.
// Phase 2: with chunks set, computes shard geometry from min_shard_bytes and
//   target_concurrent_shards (see dims_set_shard_geometry).
// Cross-phase: checks that chunks_per_shard_total <= MAX_PARTS_PER_SHARD.
//   If violated, halves the chunk target and retries. Bails when the target
//   would drop below min_chunk_bytes.
//
// shard_alignment: 0 = no alignment constraint.
// min_chunk_bytes: floor on per-chunk bytes; 0 = no floor (clamped to bpe).
// diag: optional out-param describing the failure reason and relevant context
//   when the solver returns non-zero; caller may pass NULL.
// Modifies config->dimensions in place (chunk_size and chunks_per_shard) and
// config->epochs_per_batch (set to the chosen K on success).
// Returns 0 on success.
int
tile_stream_gpu_advise_layout(struct tile_stream_configuration* config,
                              size_t target_chunk_bytes,
                              size_t min_chunk_bytes,
                              const int* ratios,
                              size_t budget_bytes,
                              size_t min_shard_bytes,
                              uint32_t target_concurrent_shards,
                              uint32_t min_append_shards,
                              size_t shard_alignment,
                              struct advise_layout_diagnostic* diag);

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

// Threads the staging-copy pool runs on, counting the caller. The pool caps
// its helpers, so this can be below the requested thread count.
int
tile_stream_gpu_worker_threads(const struct tile_stream_gpu* s);
