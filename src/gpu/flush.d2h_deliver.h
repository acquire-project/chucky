#pragma once

#include "gpu/stream.internal.h"
#include "stream/dim_info.h"

// Initialize the D2H+deliver stage. Returns 0 on success.
int
d2h_deliver_init(struct d2h_deliver_stage* stage,
                 size_t shard_alignment,
                 CUstream compute);

// Destroy the D2H+deliver stage.
void
d2h_deliver_destroy(struct d2h_deliver_stage* stage);

// Enqueue the full D2H (offset + bulk) for this batch on the d2h stream,
// non-blocking. Briefly host-syncs on the offset D2H to size the bulk
// transfer, and waits on prior sink IO fences for this fc. ready[fc] is
// recorded after bulk D2H completes. Returns 0 on success.
int
d2h_deliver_kick(struct d2h_deliver_stage* stage,
                 const struct flush_handoff* handoff,
                 const struct level_geometry* levels,
                 const struct batch_state* batch,
                 const struct dim_info* dims,
                 struct shard_sink* sink,
                 CUstream d2h_stream);

// Synchronize D2H, record metrics, deliver to sinks.
// Returns writer_ok() on success. d2h_stream is used to dispatch the
// exact-size bulk D2H once h_offsets/h_permuted_sizes have landed.
struct writer_result
d2h_deliver_drain(struct d2h_deliver_stage* stage,
                  const struct flush_handoff* handoff,
                  const struct level_geometry* levels,
                  const struct batch_state* batch,
                  const struct dim_info* dims,
                  const struct tile_stream_layout* layout,
                  const struct tile_stream_configuration* config,
                  struct shard_sink* sink,
                  const struct lod_state* lod,
                  const struct lod_shared_state* lod_shared,
                  struct stream_metrics* metrics,
                  struct platform_clock* metadata_update_clock,
                  CUstream d2h_stream);
