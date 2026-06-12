#pragma once

// Declared cross-stream / host-stream ordering edges for the GPU pipeline
// (docs/gpu-orchestration.md). Every cuEventRecord/cuStreamWaitEvent/
// cuStreamWaitValue64/cuEventQuery pair that orders work between streams (or
// between the host and a stream) goes through this table; an edge that is
// not declared here must be timing-only.
//
// Timing-only events are excluded (they must not masquerade as ordering):
//   staging t_h2d_start/t_scatter_start; compress t_compress_start/end;
//   d2h t_d2h_start; lod timing t_start/t_scatter_end/t_reduce_end/
//   t_append_end (t_end doubles as GPU_EDGE_LOD_DONE and is bound here).

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
  GPU_STREAM_DRAIN, // gpu_streams.drain
  GPU_STREAM_ID_COUNT,
};

enum gpu_edge_kind
{
  GPU_EDGE_EVENT,       // CUevent record/wait
  GPU_EDGE_GEN_COUNTER, // host-published generation + cuStreamWaitValue64 GEQ
  GPU_EDGE_HOST_RULE,   // host call-order invariant; no GPU primitive
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
  GPU_EDGE_POOL_FILLED, // compute -> compress: chunk-pool batch contents
                        // (single instance; one batch in flight at record)
  GPU_EDGE_LOD_DONE,    // compute -> compress: LOD chunks in pool
                        // (bound to lod_shared timing t_end; multiscale only)
  GPU_EDGE_AGG_DONE,    // compress -> d2h: aggregate slot outputs ready
  GPU_EDGE_POOL_CONSUMED, // compress -> compute (alias of AGG_DONE):
                          // pool buf[fc] reuse / re-zero (#140)
  GPU_EDGE_SLOT_DRAINED,  // d2h|drain -> compress: agg slot reuse
  GPU_EDGE_D2H_DONE,      // d2h|drain -> HOST (alias of SLOT_DRAINED):
                          // h_aggregated stable for sink delivery
  GPU_EDGE_CHUNK_INDEX_READY, // d2h -> HOST: h_offsets/h_permuted_sizes
                              // landed; drain copy source stable

  // Tail-generation gate (#142).
  GPU_EDGE_TAIL_PUBLISHED, // HOST -> compress: d_tail_bytes/d_tail_carry
                           // generation k published by kick k's delivery

  // Host call-order invariants (debug-asserted; no GPU primitive).
  GPU_EDGE_DRAIN_BEFORE_REKICK,  // drain pending[fc] before re-kicking fc
  GPU_EDGE_DELIVER_OLDEST_FIRST, // drains follow kick order (tail gate GEQ
                                 // relies on this)

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
};

const struct gpu_edge_desc*
gpu_edge_describe(enum gpu_edge e);

struct gpu_edge_state
{
  CUevent ev[2];
  uint8_t owned[2];
  struct stream_metric* stall; // host-poll stall accumulation; may be NULL
#ifndef NDEBUG
  uint64_t records[2]; // edge_record/publish calls (seeds excluded)
  uint64_t waits[2];   // edge_wait/host_wait/wait_gen calls
#endif
};

struct gpu_ordering
{
  struct gpu_edge_state edge[GPU_EDGE_COUNT];
  CUstream streams[GPU_STREAM_ID_COUNT]; // registered for debug decl checks

  // GEN_COUNTER runtime (GPU_EDGE_TAIL_PUBLISHED). Pinned + device-mapped;
  // absent (NULL/0) when the gate was never initialized or could not be
  // allocated/mapped — the lazy flush path then host-drains instead
  // (SCHEDULE_DRAIN_BEFORE_KICK, schedule.h).
  volatile uint64_t* h_tail_seq_flag;
  CUdeviceptr d_tail_seq;
  uint64_t kick_seq; // kicks enqueued (gate threshold)
  uint64_t tail_seq; // uploads published (host mirror)
  int tail_gate_supported;
};

// Create + seed owned events on seed_stream. External edges stay unbound
// until gpu_ordering_bind.
int
gpu_ordering_init(struct gpu_ordering* ord, CUstream seed_stream);

// Destroys owned events and the gate counter. Debug builds log a dead-edge
// warning for declared edges that were recorded but never waited this run.
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
gpu_ordering_bind(struct gpu_ordering* ord,
                  enum gpu_edge e,
                  int i,
                  CUevent ev);

// The edge's event (alias edges resolve to their owner's). NULL if unbound.
CUevent
gpu_ordering_event(const struct gpu_ordering* ord, enum gpu_edge e, int i);

int
gpu_edge_record(struct gpu_ordering* ord,
                enum gpu_edge e,
                int i,
                CUstream stream);

int
gpu_edge_wait(struct gpu_ordering* ord, enum gpu_edge e, int i, CUstream stream);

// Host-side poll (cuEventQuery loop). Returns 0 on completion — including
// CUDA_ERROR_DEINITIALIZED at context teardown, where exiting cleanly is
// correct — and 1 on any other error. Blocked time accrues to the edge's
// stall metric when attached.
int
gpu_edge_host_wait(struct gpu_ordering* ord, enum gpu_edge e, int i);

void
gpu_ordering_attach_stall_metric(struct gpu_ordering* ord,
                                 enum gpu_edge e,
                                 struct stream_metric* m);

// --- GEN_COUNTER (tail gate) ---

// Allocate + map the pinned counter and probe cuStreamWaitValue64 support
// with an already-satisfied wait on probe_stream. Degrades gracefully:
// alloc/map failure or CUDA_ERROR_NOT_SUPPORTED leaves the gate off
// (gate_supported 0) and returns 0; only unexpected probe errors return 1.
int
gpu_ordering_gate_init(struct gpu_ordering* ord, CUstream probe_stream);

static inline int
gpu_ordering_gate_supported(const struct gpu_ordering* ord)
{
  return ord->tail_gate_supported;
}

static inline int
gpu_ordering_gate_active(const struct gpu_ordering* ord)
{
  return ord->h_tail_seq_flag != NULL;
}

// Queue a wait for the published generation to reach this kick's threshold,
// then advance the threshold. The threshold advances even when enable is 0:
// every kick is drained (published) exactly once, so the count stays the
// next kick's threshold. No-op when the gate was never initialized.
int
gpu_edge_wait_gen(struct gpu_ordering* ord,
                  enum gpu_edge e,
                  CUstream stream,
                  int enable);

// Publish the drained kick's generation. Must run exactly once per drain,
// on failure exits too — a skipped publish leaves the gate unsatisfiable.
void
gpu_edge_publish(struct gpu_ordering* ord, enum gpu_edge e);

// Satisfy every gate wait ever enqueued. Required before any blocking
// stream/context sync when a kick may have been left undrained. kick_seq
// satisfies every threshold; UINT64_MAX would not (CU_STREAM_WAIT_VALUE_GEQ
// is a signed ring compare).
void
gpu_edge_release_all(struct gpu_ordering* ord);

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
  } while (0)
#endif
