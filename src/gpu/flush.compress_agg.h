#pragma once

#include "gpu/stream.internal.h"

struct computed_stream_layouts;

int
compress_agg_init(struct compress_agg_stage* stage,
                  const struct computed_stream_layouts* cl,
                  const struct tile_stream_configuration* config,
                  CUstream compute);

void
compress_agg_destroy(struct compress_agg_stage* stage, int nlod);

// Satisfy every tail-gate wait ever enqueued. Required before any blocking
// stream/context sync when a kick may have been left undrained (failed
// flush): its parked wait would otherwise never complete.
void
compress_agg_release_tail_gate(struct compress_agg_stage* stage);

int
compress_agg_kick(struct compress_agg_stage* stage,
                  const struct compress_agg_input* in,
                  const struct level_geometry* levels,
                  CUstream compress_stream,
                  struct flush_handoff* out);
