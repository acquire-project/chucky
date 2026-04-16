#pragma once

#include "multiarray/multiarray.h"
#include "types.stream.h"

struct multiarray_tile_stream_gpu;
struct stream_metrics;

// Create a GPU multiarray stream.  GPU resources (pools, staging, codec) are
// shared across all arrays and sized to the maximum requirement, so memory
// usage is constant with respect to n_arrays.  Only one array may be active
// at a time; switching requires an epoch boundary.
//
// enable_metrics is currently ignored on the GPU path: metrics are always
// collected (CUDA events are required for stream synchronization regardless).
// Accepted for API symmetry with the CPU multiarray constructor.
struct multiarray_tile_stream_gpu*
multiarray_tile_stream_gpu_create(
  int n_arrays,
  const struct tile_stream_configuration configs[],
  struct shard_sink* sinks[],
  int enable_metrics);

void
multiarray_tile_stream_gpu_destroy(struct multiarray_tile_stream_gpu* ms);

struct multiarray_writer*
multiarray_tile_stream_gpu_writer(struct multiarray_tile_stream_gpu* ms);

struct stream_metrics
multiarray_tile_stream_gpu_get_metrics(
  const struct multiarray_tile_stream_gpu* ms);
