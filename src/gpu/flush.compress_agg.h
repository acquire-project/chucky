#pragma once

#include "stream.internal.h"

struct computed_stream_layouts;

// Initialize the compress+aggregate stage. Returns 0 on success.
int
compress_agg_init(struct compress_agg_stage* stage,
                  const struct computed_stream_layouts* cl,
                  const struct tile_stream_configuration* config,
                  CUstream compute);

// Destroy the compress+aggregate stage.
void
compress_agg_destroy(struct compress_agg_stage* stage, int nlod);

// Kick compress+aggregate for a batch. Populates handoff for D2H stage.
// Returns 0 on success.
int
compress_agg_kick(struct compress_agg_stage* stage,
                  const struct compress_agg_input* in,
                  const struct level_geometry* levels,
                  const struct batch_state* batch,
                  CUstream compress_stream,
                  struct flush_handoff* out);
