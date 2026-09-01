#pragma once

#include "gpu/stream.internal.h"
#include "stream/dim_info.h"

// payload_copy_stream must not be the D2H metadata stream because exact-size
// payload copies are dispatched only after the metadata copies host-complete.
int
d2h_deliver_init(struct d2h_deliver_stage* stage,
                 size_t shard_alignment,
                 enum aggregate_size_kind size_kind,
                 struct gpu_ordering* ordering,
                 CUstream payload_copy_stream,
                 CUstream compute);

void
d2h_deliver_destroy(struct d2h_deliver_stage* stage);

// Sink delivery for the host-complete batch. CUDA copy planning and readiness
// live behind host_batch_copy; delivery deliberately stays outside it.
struct writer_result
d2h_deliver_host_batch(struct d2h_deliver_stage* stage,
                       const struct flush_handoff* handoff,
                       struct host_batch* host,
                       CUevent payload_start,
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
