#include "sink_throttled.h"
#include "platform/platform.h"
#include "util/prelude.h"

#include <stdlib.h>

// --- Throttled shard_sink: synthetic IO pressure for measurement ---
//
// Same async shape as shard_pool_fs: owns its own io_queue, worker sleeps
// proportional to bytes+latency instead of doing real IO. Exercises the
// record_fence/wait_fence/pending_bytes path so stall timing in the GPU
// flush pipeline can be measured on machines where the real GPU can't
// saturate disk IO.

struct throttled_job
{
  uint64_t nbytes;
  uint64_t latency_ns;
  uint64_t bytes_per_sec;
  _Atomic uint64_t* retired_bytes;
  _Atomic uint64_t* total_bytes;
};

static void
throttled_fn(void* arg)
{
  struct throttled_job* j = (struct throttled_job*)arg;
  int64_t ns = (int64_t)j->latency_ns;
  if (j->bytes_per_sec > 0)
    ns += (int64_t)((j->nbytes * 1000000000ull) / j->bytes_per_sec);
  if (ns > 0)
    platform_sleep_ns(ns);
  atomic_fetch_add(j->retired_bytes, j->nbytes);
  atomic_fetch_add(j->total_bytes, j->nbytes);
}

static int
throttled_post(struct throttled_shard_sink* s, size_t nbytes)
{
  struct throttled_job* j = (struct throttled_job*)malloc(sizeof(*j));
  CHECK(Error, j);
  j->nbytes = nbytes;
  j->latency_ns = s->latency_ns;
  j->bytes_per_sec = s->bytes_per_sec;
  j->retired_bytes = &s->retired_bytes;
  j->total_bytes = &s->total_bytes;
  if (io_queue_post(s->queue, throttled_fn, j, free)) {
    free(j);
    goto Error;
  }
  s->queued_bytes += nbytes;
  return 0;

Error:
  return 1;
}

static int
throttled_shard_write(struct shard_writer* self,
                      uint64_t offset,
                      const void* beg,
                      const void* end)
{
  (void)offset;
  struct throttled_shard_writer* w = (struct throttled_shard_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  return throttled_post(w->parent, nbytes);
}

static int
throttled_shard_write_direct(struct shard_writer* self,
                             uint64_t offset,
                             const void* beg,
                             const void* end)
{
  (void)offset;
  struct throttled_shard_writer* w = (struct throttled_shard_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  return throttled_post(w->parent, nbytes);
}

static int
throttled_shard_finalize(struct shard_writer* self)
{
  (void)self;
  return 0;
}

static struct shard_writer*
throttled_shard_open(struct shard_sink* self,
                     uint8_t level,
                     uint64_t shard_index)
{
  (void)level;
  (void)shard_index;
  struct throttled_shard_sink* s = (struct throttled_shard_sink*)self;
  return &s->writer.base;
}

static struct io_event
throttled_shard_record_fence(struct shard_sink* self, uint8_t level)
{
  (void)level;
  struct throttled_shard_sink* s = (struct throttled_shard_sink*)self;
  return io_queue_record(s->queue);
}

static void
throttled_shard_wait_fence(struct shard_sink* self,
                           uint8_t level,
                           struct io_event ev)
{
  (void)level;
  struct throttled_shard_sink* s = (struct throttled_shard_sink*)self;
  io_event_wait(s->queue, ev);
}

static int
throttled_shard_has_error(const struct shard_sink* self)
{
  (void)self;
  return 0;
}

static size_t
throttled_shard_pending_bytes(const struct shard_sink* self)
{
  const struct throttled_shard_sink* s =
    (const struct throttled_shard_sink*)self;
  uint64_t retired = atomic_load(&s->retired_bytes);
  if (s->queued_bytes <= retired)
    return 0;
  return (size_t)(s->queued_bytes - retired);
}

int
throttled_shard_sink_init(struct throttled_shard_sink* s,
                          uint64_t io_bw_mbps,
                          uint64_t io_latency_us)
{
  *s = (struct throttled_shard_sink){ 0 };
  s->latency_ns = io_latency_us * 1000ull;
  s->bytes_per_sec = io_bw_mbps * 1024ull * 1024ull;
  s->queue = io_queue_create();
  if (!s->queue)
    return 1;

  s->base.open = throttled_shard_open;
  s->base.record_fence = throttled_shard_record_fence;
  s->base.wait_fence = throttled_shard_wait_fence;
  s->base.has_error = throttled_shard_has_error;
  s->base.pending_bytes = throttled_shard_pending_bytes;

  s->writer = (struct throttled_shard_writer){
    .base = { .write = throttled_shard_write,
              .write_direct = throttled_shard_write_direct,
              .finalize = throttled_shard_finalize },
    .parent = s,
  };
  return 0;
}

void
throttled_shard_sink_teardown(struct throttled_shard_sink* s)
{
  if (s->queue) {
    io_event_wait(s->queue, io_queue_record(s->queue));
    io_queue_destroy(s->queue);
  }
  *s = (struct throttled_shard_sink){ 0 };
}
