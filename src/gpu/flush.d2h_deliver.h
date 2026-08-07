#pragma once

#include "gpu/stream.internal.h"
#include "stream/dim_info.h"

// drain_stream is borrowed; it must not be the d2h stream because exact-size
// payload copies are dispatched only after the metadata copies host-complete.
int
d2h_deliver_init(struct d2h_deliver_stage* stage,
                 size_t shard_alignment,
                 struct gpu_ordering* ord,
                 CUstream drain_stream,
                 CUstream compute);

void
d2h_deliver_destroy(struct d2h_deliver_stage* stage);

// Payload phases over an already-acquired slot; the acquires and releases
// around them are placed by the schedule (schedule_d2h_kick /
// schedule_d2h_drain).

// Chunk-index copies. Pass-through codecs also run the bulk D2H here so it
// overlaps the next batch's compute; compressed codecs land the chunk index
// only and size the bulk copies at drain time.
int
d2h_deliver_kick(struct d2h_deliver_stage* stage,
                 const struct flush_handoff* handoff,
                 struct aggregate_slot* slot,
                 CUstream d2h_stream);

// Compressed-only: per-LOD bulk copies sized from the landed chunk index.
int
d2h_deliver_drain_copy(struct d2h_deliver_stage* stage,
                       const struct flush_handoff* handoff,
                       struct aggregate_slot* slot);

// Sink delivery + tail-state upload for the host-complete slot.
struct writer_result
d2h_deliver_drain_sink(struct d2h_deliver_stage* stage,
                       const struct flush_handoff* handoff,
                       struct aggregate_slot* slot,
                       struct compress_agg_array* shards,
                       const struct level_geometry* levels,
                       const struct tile_stream_layout* layout,
                       const struct tile_stream_configuration* config,
                       struct shard_sink* sink,
                       struct stream_metrics* metrics);

int
d2h_deliver_update_metadata(const struct flush_handoff* handoff,
                            const struct dim_info* dims_info,
                            const struct tile_stream_configuration* config,
                            struct shard_sink* sink,
                            struct platform_clock* metadata_update_clock);
