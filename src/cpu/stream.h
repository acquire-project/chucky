#pragma once

#include "stream.config.h"
#include "types.stream.h"

struct tile_stream_cpu;

// Create a CPU streaming pipeline. Returns NULL on failure or f16 dtype.
// The config->dimensions pointer must remain valid for the lifetime of the stream.
struct tile_stream_cpu*
tile_stream_cpu_create(const struct tile_stream_configuration* config);

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

struct tile_stream_cpu_memory_info
{
  size_t heap_bytes;             // total
  size_t chunk_pool_bytes;       // total_chunks * chunk_stride * bpe
  size_t compressed_pool_bytes;  // total_chunks * max_output_size
  size_t comp_sizes_bytes;       // total_chunks * sizeof(size_t)
  size_t aggregate_bytes;        // shared ws + per-level perm
  size_t lod_bytes;              // linear + lod_values + morton_lut + batch_offsets + dim0
  size_t shard_bytes;            // active_shard arrays + index buffers

  uint64_t chunks_per_epoch;
  uint64_t total_chunks;
  size_t max_output_size;
  int nlod;
  uint32_t epochs_per_batch;
};

int tile_stream_cpu_memory_estimate(
  const struct tile_stream_configuration* config,
  struct tile_stream_cpu_memory_info* info);

// Find the largest power-of-2 chunk size (starting from target_chunk_bytes)
// that fits within budget_bytes of CPU heap memory.
// Modifies config->dimensions in place. Returns 0 on success.
int
tile_stream_cpu_advise_chunk_sizes(
    struct tile_stream_configuration* config,
    size_t target_chunk_bytes,
    const uint8_t* ratios,
    size_t budget_bytes);
