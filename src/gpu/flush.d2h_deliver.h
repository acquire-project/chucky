#pragma once

#include "gpu/stream.internal.h"
#include "stream/dim_info.h"

// drain_stream is borrowed; it must not be the d2h stream because exact-size
// payload copies are dispatched only after the metadata copies host-complete.
int
d2h_deliver_init(struct d2h_deliver_stage* stage,
                 size_t shard_alignment,
                 enum device_aggregate_extent_kind extent_kind,
                 struct gpu_ordering* ord,
                 CUstream drain_stream,
                 CUstream compute);

void
d2h_deliver_destroy(struct d2h_deliver_stage* stage);

// Sink delivery + tail-state upload for the host-complete batch.  CUDA copy
// planning and readiness live behind d2h_materializer; delivery deliberately
// stays outside that boundary.
struct writer_result
d2h_deliver_drain_sink(struct d2h_deliver_stage* stage,
                       const struct flush_handoff* handoff,
                       struct host_batch* host,
                       CUevent payload_start,
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
