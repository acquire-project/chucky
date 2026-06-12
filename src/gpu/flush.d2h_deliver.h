#pragma once

#include "gpu/stream.internal.h"
#include "stream/dim_info.h"

int
d2h_deliver_init(struct d2h_deliver_stage* stage,
                 size_t shard_alignment,
                 struct gpu_ordering* ord,
                 CUstream compute);

void
d2h_deliver_destroy(struct d2h_deliver_stage* stage);

// Pass-through codecs complete the D2H here; compressed codecs only land
// the chunk index and finish in drain (bulk D2H is sized by actual bytes).
int
d2h_deliver_kick(struct d2h_deliver_stage* stage,
                 const struct flush_handoff* handoff,
                 struct shard_sink* sink,
                 CUstream d2h_stream);

struct writer_result
d2h_deliver_drain(struct d2h_deliver_stage* stage,
                  const struct flush_handoff* handoff,
                  const struct level_geometry* levels,
                  const struct dim_info* dims,
                  const struct tile_stream_layout* layout,
                  const struct tile_stream_configuration* config,
                  struct shard_sink* sink,
                  const struct lod_state* lod,
                  const struct lod_shared_state* lod_shared,
                  struct stream_metrics* metrics,
                  struct platform_clock* metadata_update_clock,
                  CUstream d2h_stream);
