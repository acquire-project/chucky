#pragma once

#include "gpu/stream.internal.h"

struct computed_stream_layouts;
struct engine_limits;

// Shared (maxima-sized) stage resources: codec, compressed buffers, unified
// aggregate slots, LUTs, per-shard device tables.
int
compress_agg_init_shared(struct compress_agg_stage* stage,
                         const struct engine_limits* lim,
                         struct codec_config codec,
                         size_t typesize,
                         struct gpu_ordering* ord,
                         CUstream compute);

void
compress_agg_destroy_shared(struct compress_agg_stage* stage);

// Per-array slice: aggregate layouts, shard_state, and tail buffers.
int
compress_agg_array_init(struct compress_agg_array* ar,
                        const struct computed_stream_layouts* cl);

void
compress_agg_array_destroy(struct compress_agg_array* ar);

int
compress_agg_host_output_requirements(
  const struct computed_stream_layouts* cl,
  const struct tile_stream_configuration* config,
  size_t* output_bytes);

int
compress_agg_array_init_output(struct compress_agg_array* ar,
                               size_t output_bytes);

struct host_output_pool*
compress_agg_output_pool_create(size_t output_bytes);

// Single-array convenience: shared + array init from one layout, with the
// shard-capacity table uploaded.
int
compress_agg_init(struct compress_agg_stage* stage,
                  const struct computed_stream_layouts* cl,
                  const struct tile_stream_configuration* config,
                  struct gpu_ordering* ord,
                  CUstream compute);

void
compress_agg_destroy(struct compress_agg_stage* stage);

// Sizing mirror of compress_agg_init_shared + compress_agg_array_init for
// one array, for the memory estimate.
int
compress_agg_memory_estimate(const struct engine_limits* lim,
                             const struct computed_stream_layouts* cl,
                             struct codec_config codec,
                             size_t* compressed_pool_bytes,
                             size_t* codec_bytes,
                             size_t* aggregate_device_bytes,
                             size_t* aggregate_host_bytes);

// Host-side batch planning plus LUT/shard-table uploads. No pool access.
int
compress_agg_prepare(struct compress_agg_stage* stage,
                     const struct compress_agg_input* in,
                     const struct level_geometry* levels,
                     CUstream compress_stream,
                     struct compress_agg_plan* plan);

// Compress the batch out of the acquired pool view (CODEC_NONE records the
// timing events only). Reads no tail state.
int
compress_agg_compress(struct compress_agg_stage* stage,
                      const struct compress_agg_input* in,
                      const struct level_geometry* levels,
                      struct gpu_pool_view pool_buf,
                      CUstream compress_stream);

// Aggregate into the acquired slot.
int
compress_agg_aggregate(struct compress_agg_stage* stage,
                       const struct compress_agg_plan* plan,
                       int fc,
                       struct aggregate_slot* slot,
                       struct gpu_pool_view pool_buf,
                       CUstream compress_stream);

void
compress_agg_fill_handoff(struct compress_agg_stage* stage,
                          const struct compress_agg_input* in,
                          const struct compress_agg_plan* plan,
                          struct flush_handoff* out);
