#pragma once

#include "gpu/stream.internal.h"

struct computed_stream_layouts;
struct engine_limits;

// Shared (maxima-sized) stage resources: codec, compressed buffers, unified
// aggregate slots, LUTs, per-shard device tables.
int
compress_agg_init_shared(struct compress_agg_stage* stage,
                         const struct engine_limits* lim,
                         enum compression_codec codec_id,
                         struct gpu_ordering* ord,
                         CUstream compute);

void
compress_agg_destroy_shared(struct compress_agg_stage* stage);

// Per-array slice: aggregate layouts, shard_state, tail buffers. gate_ord
// arms the tail-generation gate when page-aligned; NULL skips it (multiarray
// sync-flush path host-orders the tail uploads via immediate drains).
int
compress_agg_array_init(struct compress_agg_array* ar,
                        const struct computed_stream_layouts* cl,
                        struct gpu_ordering* gate_ord,
                        CUstream gate_stream);

void
compress_agg_array_destroy(struct compress_agg_array* ar);

// Single-array convenience: shared + array init from one layout, with the
// tail gate armed and the shard-capacity table uploaded.
int
compress_agg_init(struct compress_agg_stage* stage,
                  const struct computed_stream_layouts* cl,
                  const struct tile_stream_configuration* config,
                  struct gpu_ordering* ord,
                  CUstream compute);

void
compress_agg_destroy(struct compress_agg_stage* stage);

int
compress_agg_kick(struct compress_agg_stage* stage,
                  const struct compress_agg_input* in,
                  const struct level_geometry* levels,
                  CUstream compress_stream,
                  struct flush_handoff* out);
