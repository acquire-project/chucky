#pragma once

#include "stream.internal.h"

// Initialize the D2H+deliver stage. Returns 0 on success.
// `levels` is borrowed from compress_agg_stage — must outlive this stage.
int
d2h_deliver_init(struct d2h_deliver_stage* stage,
                 struct level_flush_state* levels,
                 int nlod,
                 CUstream compute);

// Destroy the D2H+deliver stage.
void
d2h_deliver_destroy(struct d2h_deliver_stage* stage);

// Enqueue D2H transfers (async). Waits on IO fences first.
// Returns 0 on success.
int
d2h_deliver_kick(struct d2h_deliver_stage* stage,
                 const struct flush_handoff* handoff,
                 const struct level_geometry* levels,
                 const struct batch_state* batch,
                 const struct tile_stream_configuration* config,
                 struct shard_sink* sink,
                 CUstream d2h_stream);

// Synchronize D2H, record metrics, deliver to sinks.
// Returns writer_ok() on success.
struct writer_result
d2h_deliver_drain(struct d2h_deliver_stage* stage,
                  const struct flush_handoff* handoff,
                  const struct level_geometry* levels,
                  const struct batch_state* batch,
                  const struct tile_stream_layout* layout,
                  const struct tile_stream_configuration* config,
                  struct shard_sink* sink,
                  const struct lod_state* lod,
                  struct stream_metrics* metrics,
                  struct platform_clock* metadata_update_clock);
