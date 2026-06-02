#pragma once

#include "gpu/stream.internal.h"

struct computed_stream_layouts;

struct compress_agg_work
{
  int fc;
  int active_output_idx;
  const void* d_aggregate_src;
  struct aggregate_slot* scratch_slot;
  struct batch_aggregate_layout layout;
  uint32_t per_lod_n_active[LOD_MAX_LEVELS];
  size_t page_size;
};

int
compress_agg_init(struct compress_agg_stage* stage,
                  const struct computed_stream_layouts* cl,
                  const struct tile_stream_configuration* config,
                  CUstream compute);

void
compress_agg_destroy(struct compress_agg_stage* stage, int nlod);

int
compress_agg_measure(struct compress_agg_stage* stage,
                     const struct compress_agg_input* in,
                     const struct level_geometry* levels,
                     CUstream compress_stream,
                     struct flush_handoff* out,
                     struct compress_agg_work* work);

int
compress_agg_write_reserved(struct compress_agg_stage* stage,
                            const struct compress_agg_work* work,
                            const struct output_slot_reservation* reservation,
                            CUstream compress_stream);
