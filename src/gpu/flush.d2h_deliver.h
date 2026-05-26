#pragma once

#include "gpu/stream.internal.h"
#include "stream/dim_info.h"

int
d2h_deliver_init(struct d2h_deliver_stage* stage,
                 size_t shard_alignment,
                 CUstream compute);

void
d2h_deliver_destroy(struct d2h_deliver_stage* stage);

// Enqueue the offset/permuted-sizes D2H and (passthrough only) the bulk
// data D2H, non-blocking on d2h_stream. Records h_chunk_index_ready[fc]
// after the chunk-index D2H; for passthrough, records ready[fc] and
// slot->ready after the bulk dispatch. Compressed paths defer the bulk
// D2H to d2h_deliver_drain since exact-size requires the chunk index.
// Returns 0 on success.
int
d2h_deliver_kick(struct d2h_deliver_stage* stage,
                 const struct flush_handoff* handoff,
                 const struct level_geometry* levels,
                 const struct batch_state* batch,
                 const struct dim_info* dims,
                 struct shard_sink* sink,
                 CUstream d2h_stream);

// Synchronize D2H, record metrics, deliver to sinks. For compressed
// batches, also dispatches the exact-size bulk D2H once h_offsets has
// landed, then records ready[fc] and slot->ready.
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
