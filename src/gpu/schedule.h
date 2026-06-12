#pragma once

// One owner for GPU orchestration (docs/gpu-orchestration.md §4): stream
// creation, pipeline depth, which stages run for the active configuration,
// and degraded schedules. Stages stay payload; the schedule places the
// cross-stage acquires/releases and decides when kicks and drains happen.

#include <cuda.h>

struct gpu_ordering;

struct gpu_streams
{
  CUstream h2d, compute, compress, d2h;
  // Drain-time copies must not share the d2h stream: by drain time it can
  // already hold the next kick's GPU_EDGE_AGG_DONE wait, which the tail
  // gate keeps parked until that drain publishes — sharing would deadlock.
  CUstream drain;
};

int
gpu_streams_init(struct gpu_streams* s);

void
gpu_streams_destroy(struct gpu_streams* s);

// Lets debug builds check each edge's record/wait stream against its
// declaration.
void
gpu_streams_register(const struct gpu_streams* s, struct gpu_ordering* ord);
