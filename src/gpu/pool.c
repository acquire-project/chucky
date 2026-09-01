#include "gpu/pool.h"

#include <assert.h>
#include <string.h>

// Edge instances are declared in the table (per_fc), not chosen by callers:
// single-instance edges fold every slot onto instance 0.
static int
inst(enum gpu_edge e, int slot)
{
  return gpu_edge_describe(e)->per_fc ? slot : 0;
}

void
gpu_pool_init(struct gpu_pool* p,
              struct gpu_ordering* ord,
              enum gpu_edge ready,
              enum gpu_edge consumed)
{
  memset(p, 0, sizeof(*p));
  p->ord = ord;
  p->ready = ready;
  p->consumed = consumed;
}

void
gpu_pool_bind(struct gpu_pool* p, int slot, void* payload)
{
  p->payload[slot] = payload;
}

int
gpu_pool_acquire_produce(struct gpu_pool* p,
                         int slot,
                         CUstream stream,
                         struct gpu_pool_view* out)
{
  if (p->consumed != GPU_EDGE_COUNT &&
      gpu_edge_wait(p->ord, p->consumed, inst(p->consumed, slot), stream))
    return 1;
  if (out)
    out->p = p->payload[slot];
  return 0;
}

int
gpu_pool_release_produce(struct gpu_pool* p, int slot, CUstream stream)
{
  assert(p->ready != GPU_EDGE_COUNT);
  return gpu_edge_record(p->ord, p->ready, inst(p->ready, slot), stream);
}

int
gpu_pool_acquire_consume(struct gpu_pool* p,
                         int slot,
                         CUstream stream,
                         struct gpu_pool_view* out)
{
  if (p->ready != GPU_EDGE_COUNT &&
      gpu_edge_wait(p->ord, p->ready, inst(p->ready, slot), stream))
    return 1;
  if (out)
    out->p = p->payload[slot];
  return 0;
}

int
gpu_pool_release_consume(struct gpu_pool* p, int slot, CUstream stream)
{
  assert(p->consumed != GPU_EDGE_COUNT);
  return gpu_edge_record(p->ord, p->consumed, inst(p->consumed, slot), stream);
}

int
gpu_pool_host_acquire_produce(struct gpu_pool* p,
                              int slot,
                              struct gpu_pool_view* out)
{
  if (p->consumed != GPU_EDGE_COUNT &&
      gpu_edge_host_wait(p->ord, p->consumed, inst(p->consumed, slot)))
    return 1;
  if (out)
    out->p = p->payload[slot];
  return 0;
}

int
gpu_pool_host_acquire_consume(struct gpu_pool* p,
                              int slot,
                              struct gpu_pool_view* out)
{
  if (p->ready != GPU_EDGE_COUNT &&
      gpu_edge_host_wait(p->ord, p->ready, inst(p->ready, slot)))
    return 1;
  if (out)
    out->p = p->payload[slot];
  return 0;
}

int
gpu_pool_host_acquire_consume_split(struct gpu_pool* p,
                                    int slot,
                                    enum gpu_edge prerequisite,
                                    struct stream_metric* before,
                                    struct stream_metric* after,
                                    struct gpu_pool_view* out)
{
  if (p->ready != GPU_EDGE_COUNT &&
      gpu_edge_host_wait_split(
        p->ord, p->ready, prerequisite, inst(p->ready, slot), before, after))
    return 1;
  if (out)
    out->p = p->payload[slot];
  return 0;
}

struct gpu_pool_view
gpu_pool_at(struct gpu_pool* p, int slot, size_t byte_offset)
{
  return (struct gpu_pool_view){ .p = (char*)p->payload[slot] + byte_offset };
}
