#pragma once

#include "types.stream.h"
#include "writer.h"

struct tile_stream_layout;
struct tile_stream_cpu;

// Create a CPU streaming pipeline. Returns NULL on failure or f16 dtype.
// The config->dimensions pointer must remain valid for the lifetime of the
// stream.
struct tile_stream_cpu*
tile_stream_cpu_create(const struct tile_stream_configuration* config,
                       struct shard_sink* sink);

void
tile_stream_cpu_destroy(struct tile_stream_cpu* s);

struct stream_metrics
tile_stream_cpu_get_metrics(const struct tile_stream_cpu* s);

const struct tile_stream_layout*
tile_stream_cpu_layout(const struct tile_stream_cpu* s);

struct writer*
tile_stream_cpu_writer(struct tile_stream_cpu* s);

uint64_t
tile_stream_cpu_cursor(const struct tile_stream_cpu* s);

// Threads the pipeline pool runs on, counting the caller. This is the pool
// max_threads sizes, so it is what a benchmark should report as the thread
// count that ran.
int
tile_stream_cpu_worker_threads(const struct tile_stream_cpu* s);

struct tile_stream_cpu_memory_info
{
  size_t heap_bytes;            // total
  size_t chunk_pool_bytes;      // K * total_chunks * chunk_stride * bpe
  size_t compressed_pool_bytes; // K * total_chunks * max_output_size
  size_t comp_sizes_bytes;      // K * total_chunks * sizeof(size_t)
  size_t aggregate_bytes; // 2x per-batch slots (data + perm + gather + sizes
                          // + offsets + chunk_sizes) + batch_active_masks +
                          // pool_epochs_scratch
  size_t lod_bytes;       // linear + lod_values + morton_lut + batch_offsets +
                          // append_accum
  size_t shard_bytes;     // active_shard arrays + index buffers

  uint64_t chunks_per_epoch;
  uint64_t total_chunks;
  size_t max_output_size;
  int nlod;
  uint32_t epochs_per_batch;
};

// shard_alignment: 0 = no alignment constraint.
int
tile_stream_cpu_memory_estimate(const struct tile_stream_configuration* config,
                                size_t shard_alignment,
                                struct tile_stream_cpu_memory_info* info);

// Solve chunk + shard layout for the CPU backend.
// See tile_stream_gpu_advise_layout for semantics.
int
tile_stream_cpu_advise_layout(struct tile_stream_configuration* config,
                              size_t target_chunk_bytes,
                              size_t min_chunk_bytes,
                              const int* ratios,
                              size_t budget_bytes,
                              size_t min_shard_bytes,
                              uint32_t target_concurrent_shards,
                              uint32_t min_append_shards,
                              size_t shard_alignment,
                              struct advise_layout_diagnostic* diag);
