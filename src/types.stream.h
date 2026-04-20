#pragma once

#include "defs.limits.h"
#include "dimension.h"
#include "dtype.h"
#include "types.codec.h"
#include "types.lod.h"

#include <stddef.h>
#include <stdint.h>

struct stream_metric
{
  const char* name;
  float ms;            // cumulative
  float best_ms;       // best single measurement (1e30f = not yet measured)
  double input_bytes;  // cumulative bytes read by stage
  double output_bytes; // cumulative bytes written by stage
  int count;
};

struct stream_metrics
{
  struct stream_metric memcpy;
  struct stream_metric h2d;
  struct stream_metric lod_gather;
  struct stream_metric lod_reduce;
  struct stream_metric lod_append_fold;
  struct stream_metric lod_morton_chunk;
  struct stream_metric scatter;
  struct stream_metric compress;
  struct stream_metric aggregate;
  struct stream_metric d2h;
  struct stream_metric sink;

  // Stall metrics — wall-clock time the host is blocked at each sync point.
  // Populated only on the GPU path.
  struct stream_metric
    flush_stall; // d2h_deliver_drain in drain_kick_and_swap (Phase B)
  struct stream_metric kick_sync_stall; // cuEventSynchronize in drain_bulk_d2h
  struct stream_metric io_fence_stall;  // wait_io_fences in d2h_deliver_drain
  struct stream_metric backpressure; // wait at epoch boundary for IO to drain
  float max_append_ms;               // longest tile_stream_gpu_append body
  size_t peak_pending_bytes;         // max sink->pending_bytes seen
};

struct tile_stream_configuration
{
  size_t buffer_capacity_bytes;
  enum dtype dtype;
  uint8_t rank;
  struct dimension* dimensions;
  struct codec_config codec;
  enum lod_reduce_method reduce_method;
  enum lod_reduce_method append_reduce_method;
  int max_nlod; // 0 = auto, N>0 = max N total levels (1 = base only)
  int preserve_aspect_ratio; // 0 = drop dims independently (default),
                             // 1 = stop when any dim reaches chunk_size
  uint32_t epochs_per_batch; // K: 0 = auto (from target_batch_bytes)
  uint32_t
    target_batch_bytes; // target uncompressed bytes per batch (default 512 MiB)
  float metadata_update_interval_s;
  size_t backpressure_bytes; // 0 = disabled; >0 = stall at epoch boundaries
                             // when sink->pending_bytes exceeds this watermark
  int max_threads;           // 0 = OpenMP default
};

struct tile_stream_status
{
  int nlod;
  int append_downsample;
  uint32_t epochs_per_batch;
  size_t max_compressed_size;
  enum dtype dtype;
  struct codec_config codec;
  size_t codec_batch_size;
  uint32_t batch_accumulated;
  int pool_current;
  int flush_pending;
};

// Why tile_stream_{gpu,cpu}_advise_layout returned non-zero.
enum advise_layout_reason
{
  ADVISE_OK = 0,
  ADVISE_INVALID_CONFIG,       // memory_estimate or shard-geometry rejected
                               // the configuration as malformed
  ADVISE_MIN_SHARD_TOO_SMALL,  // min_shard_bytes < chunk_bytes (phase 2)
  ADVISE_BUDGET_EXCEEDED,      // no (chunk, K) combination fits budget
  ADVISE_PARTS_LIMIT_EXCEEDED, // chunks_per_shard_total > MAX_PARTS_PER_SHARD
  ADVISE_CHUNK_BUDGET_INFEASIBLE, // dims_budget_chunk_bytes rejected input
                                  // (pinned dims > budget, or target < bpe)
};

// Optional diagnostic out-param for advise_layout. Caller may pass NULL.
// On failure, reason is set and the other fields describe the last iteration
// the solver tried (closest to min_chunk_bytes). Units: bytes unless noted.
struct advise_layout_diagnostic
{
  enum advise_layout_reason reason;
  size_t floor_chunk_bytes;        // effective floor: max(min_chunk_bytes, bpe)
  size_t chunk_bytes;              // per-chunk bytes at the failing iteration
  uint32_t epochs_per_batch;       // K at failure
  size_t device_bytes;             // BUDGET_EXCEEDED: memory needed at failure
                                   // (device_bytes on GPU, heap_bytes on CPU)
  size_t budget_bytes;             // caller's budget (echoed)
  uint64_t chunks_per_shard_total; // PARTS_LIMIT_EXCEEDED: observed total
  uint64_t parts_limit;            // PARTS_LIMIT_EXCEEDED: MAX_PARTS_PER_SHARD

  // Soft-constraint status (populated on success; ADVISE_OK). Caller compares
  // against their requested target/floor to detect when the solver compromised.
  uint64_t actual_concurrent_shards; // Π shards[d] for inner dims (may exceed
                                     // target_concurrent_shards — soft)
  size_t actual_shard_bytes;         // chunk_bytes · chunks_per_shard_total
                                     // (may be < min_shard_bytes — soft)
  uint8_t min_append_shards_overrode_min_shard_bytes; // 1 if both knobs were
                                                      // set (min_append_shards
                                                      // > 1 wins, floor may be
                                                      // unmet), else 0
};
