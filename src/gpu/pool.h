#pragma once

// Slotted resource pools with generation-carried ordering
// (dev/gpu-orchestration.md). A pool binds a cycled resource's per-slot
// payload to the declared edges that order its generations:
//   ready    — producer -> consumer: slot contents valid
//   consumed — consumer -> producer: slot reusable
// Acquire queues the wait that makes the payload safe in that role before
// handing the pointer out; release records that role's completion. Edge
// identity, instancing, debug asserts, and stall metrics stay in
// gpu_ordering — pools draw edges from the table, never own events.
// GPU_EDGE_COUNT in either direction means host call order covers it (a
// declared HOST_RULE), not a GPU primitive.

#include "gpu/ordering.h"

#include <stddef.h>

struct gpu_pool
{
  struct gpu_ordering* ord; // borrowed
  enum gpu_edge ready;
  enum gpu_edge consumed;
  void* payload[2]; // borrowed; device payloads stored as (void*)(uintptr_t)
};

// Pointer minted by the pool API. Stage boundaries take views (or pool +
// slot handles), never raw pointers to pooled resources.
struct gpu_pool_view
{
  void* p;
};

static inline CUdeviceptr
gpu_pool_view_d(struct gpu_pool_view v)
{
  return (CUdeviceptr)(uintptr_t)v.p;
}

void
gpu_pool_init(struct gpu_pool* p,
              struct gpu_ordering* ord,
              enum gpu_edge ready,
              enum gpu_edge consumed);

void
gpu_pool_bind(struct gpu_pool* p, int slot, void* payload);

// Producer role: acquire waits the slot's previous generation's consumed
// release on `stream`; release records this generation's ready edge.
// Aliased edges are released through their owner pool's record only.
// `out` may be NULL when only the ordering is needed.
int
gpu_pool_acquire_produce(struct gpu_pool* p,
                         int slot,
                         CUstream stream,
                         struct gpu_pool_view* out);
int
gpu_pool_release_produce(struct gpu_pool* p, int slot, CUstream stream);

// Consumer role.
int
gpu_pool_acquire_consume(struct gpu_pool* p,
                         int slot,
                         CUstream stream,
                         struct gpu_pool_view* out);
int
gpu_pool_release_consume(struct gpu_pool* p, int slot, CUstream stream);

// Host-poll acquires (the backing edge's declared consumer is HOST).
// Blocked time accrues to the edge's stall metric.
int
gpu_pool_host_acquire_produce(struct gpu_pool* p,
                              int slot,
                              struct gpu_pool_view* out);
int
gpu_pool_host_acquire_consume(struct gpu_pool* p,
                              int slot,
                              struct gpu_pool_view* out);

// Payload offset with no ordering queued. Only for host-ordered paths
// (sync flush, bind, teardown) and offsets within a generation this caller
// already acquired on the same stream.
struct gpu_pool_view
gpu_pool_at(struct gpu_pool* p, int slot, size_t byte_offset);

// GEN_COUNTER ready edge (#142 host-published generations) behind the same
// API. Consumer acquire arms the gate; the threshold advances even when
// enable is 0 — every kick drains exactly once. Producer release publishes
// the drained generation and must run exactly once per drain, on failure
// paths too.
int
gpu_pool_acquire_consume_gen(struct gpu_pool* p, CUstream stream, int enable);
void
gpu_pool_release_produce_gen(struct gpu_pool* p);

// Force-satisfy every parked acquire. Required before any blocking stream/
// context sync when a generation may be undrained (#142 teardown). Releases
// through the whole ordering table, not just this pool's edges.
void
gpu_pool_release_all(struct gpu_pool* p);
