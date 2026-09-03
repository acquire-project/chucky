#pragma once

// Declared cross-stream / host-stream ordering edges for the GPU pipeline
// (dev/gpu-orchestration.md). Every cuEventRecord/cuStreamWaitEvent/
// cuEventQuery pair that orders work between streams (or between the host and
// a stream) goes through this table; an edge that is not declared here must be
// timing-only.
//
// Timing-only events are excluded (they must not masquerade as ordering):
//   staging t_h2d_start and the rotated scatter pairs; compress
//   t_compress_start/end; aggregate t_aggregate_start; host-copy stage
//   payload_start; lod timing t_start/t_scatter_end/t_reduce_end/
//   t_append_end/t_end.

#include <cuda.h>
#include <stdint.h>

struct stream_metric;

enum gpu_stream_id
{
  GPU_STREAM_NONE = 0,
  GPU_STREAM_HOST,
  GPU_STREAM_H2D,
  GPU_STREAM_COMPUTE,
  GPU_STREAM_COMPRESS,
  GPU_STREAM_D2H,
  GPU_STREAM_PAYLOAD_COPY, // gpu_streams.payload_copy
  GPU_STREAM_ID_COUNT,
};

enum gpu_edge_kind
{
  GPU_EDGE_EVENT,     // CUevent record/wait
  GPU_EDGE_HOST_RULE, // host call-order invariant; no GPU primitive
};

enum gpu_edge
{
  // Ingest staging (instanced by staging slot, not fc).
  GPU_EDGE_STAGING_SCATTER_DONE, // compute -> h2d: scatter/copy finished
                                 // reading d_in before H2D overwrites it
  GPU_EDGE_STAGING_H2D_DONE,     // h2d -> compute: d_in contents landed
  GPU_EDGE_STAGING_FREE,         // h2d -> HOST (alias of STAGING_H2D_DONE):
                                 // h_in safe to refill (stream.c poll)

  // Batch pipeline (instanced by fc unless noted).
  GPU_EDGE_POOL_FILLED,    // compute -> compress: chunk-pool batch contents
                           // (single instance; one batch in flight at record)
  GPU_EDGE_LOD_DONE,       // compute -> compress: LOD chunks in pool
                           // (multiscale only)
  GPU_EDGE_AGG_DONE,       // compress -> d2h: aggregate slot outputs ready
  GPU_EDGE_POOL_CONSUMED,  // compress -> compute (alias of AGG_DONE):
                           // pool buf[fc] reuse / re-zero (#140)
  GPU_EDGE_SLOT_COPY_DONE, // payload copy -> compress: aggregate slot reuse
  GPU_EDGE_D2H_DONE,       // payload copy -> HOST (alias of SLOT_COPY_DONE):
                           // leased host output stable for sink delivery
  GPU_EDGE_CHUNK_INDEX_READY, // d2h -> HOST: h_offsets/h_permuted_sizes
                              // landed; payload-copy source stable

  // Host call-order invariants (debug-asserted; no GPU primitive).
  GPU_EDGE_DELIVER_BEFORE_REKICK, // deliver pending[fc] before re-kicking fc
  GPU_EDGE_DELIVER_OLDEST_FIRST,  // delivery follows batch generation order

  GPU_EDGE_COUNT,
};

struct gpu_edge_desc
{
  const char* name;
  enum gpu_stream_id producer;
  enum gpu_stream_id producer_alt; // 0 = none
  enum gpu_stream_id consumer;
  const char* guards; // the resource this edge protects
  enum gpu_edge_kind kind;
  uint8_t per_fc;   // 1 = instanced x2 (fc or staging slot)
  uint8_t seeded;   // recorded-signaled at init
  int8_t alias_of;  // shares the owner edge's event; -1 = none
  uint8_t external; // event owned elsewhere; attached via gpu_ordering_bind
  enum gpu_stream_id consumer_alt; // 0 = none
};

const struct gpu_edge_desc*
gpu_edge_describe(enum gpu_edge e);

struct gpu_edge_state
{
  CUevent ev[2];
  uint8_t owned[2];
  struct stream_metric* stall; // host-poll stall accumulation; may be NULL
#ifndef NDEBUG
  uint64_t records[2]; // edge_record calls (seeds excluded)
  uint64_t waits[2];   // edge_wait/host_wait calls
#endif
};

struct gpu_ordering
{
  struct gpu_edge_state edge[GPU_EDGE_COUNT];
  CUstream streams[GPU_STREAM_ID_COUNT]; // registered for debug decl checks
};

// Create + seed owned events on seed_stream. External edges stay unbound
// until gpu_ordering_bind.
int
gpu_ordering_init(struct gpu_ordering* ord, CUstream seed_stream);

// Destroys owned events. Debug builds log a dead-edge warning for declared
// edges that were recorded but never waited this run.
void
gpu_ordering_destroy(struct gpu_ordering* ord);

// Optional: lets debug builds check record/wait streams against the
// declaration. Unregistered ids skip the check.
void
gpu_ordering_register_stream(struct gpu_ordering* ord,
                             enum gpu_stream_id id,
                             CUstream stream);

// Attach an externally owned (already seeded) event to an external edge.
void
gpu_ordering_bind(struct gpu_ordering* ord, enum gpu_edge e, int i, CUevent ev);

// The edge's event (alias edges resolve to their owner's). NULL if unbound.
CUevent
gpu_ordering_event(const struct gpu_ordering* ord, enum gpu_edge e, int i);

int
gpu_edge_record(struct gpu_ordering* ord,
                enum gpu_edge e,
                int i,
                CUstream stream);

int
gpu_edge_wait(struct gpu_ordering* ord,
              enum gpu_edge e,
              int i,
              CUstream stream);

// Host-side poll (cuEventQuery loop). Returns 0 on completion — including
// CUDA_ERROR_DEINITIALIZED at context teardown, where exiting cleanly is
// correct — and 1 on any other error. Blocked time accrues to the edge's
// stall metric when attached.
int
gpu_edge_host_wait(struct gpu_ordering* ord, enum gpu_edge e, int i);

// Host-poll `e` exactly as gpu_edge_host_wait does, while observing an
// upstream event to partition the blocked interval. `before` accrues until
// `prerequisite` is observed ready and `after` accrues from there until `e` is
// ready. The prerequisite query is diagnostic-only: `e` remains the ordering
// edge protecting host access, and its attached stall metric remains the
// inclusive wait.
int
gpu_edge_host_wait_split(struct gpu_ordering* ord,
                         enum gpu_edge e,
                         enum gpu_edge prerequisite,
                         int i,
                         struct stream_metric* before,
                         struct stream_metric* after);

void
gpu_ordering_attach_stall_metric(struct gpu_ordering* ord,
                                 enum gpu_edge e,
                                 struct stream_metric* m);

// --- HOST_RULE debug assert ---

#ifndef NDEBUG
void
gpu_edge_host_rule_check(struct gpu_ordering* ord,
                         enum gpu_edge e,
                         int cond,
                         const char* file,
                         int line);
#define gpu_edge_host_rule(ord, e, cond)                                       \
  gpu_edge_host_rule_check((ord), (e), (cond), __FILE__, __LINE__)
#else
#define gpu_edge_host_rule(ord, e, cond)                                       \
  do {                                                                         \
    (void)(ord);                                                               \
    (void)(cond);                                                              \
  } while (0)
#endif
