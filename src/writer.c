#include "writer.h"
#include "zarr/shard_pool.h"

#include "log/log.h"
#include "platform/platform.h"

struct writer_result
writer_append(struct writer* w, struct slice data)
{
  return w->append(w, data);
}

struct writer_result
writer_flush(struct writer* w)
{
  return w->flush(w);
}

struct writer_result
writer_append_wait(struct writer* w, struct slice data)
{
  int stalls = 0;
  const int max_stalls = 10;

  while (data.beg < data.end) {
    struct writer_result r = writer_append(w, data);
    if (r.error)
      return r;

    if (r.rest.beg == data.beg) {
      if (++stalls >= max_stalls) {
        log_error("writer_append_wait: no progress after %d retries", stalls);
        return writer_error_at(data.beg, data.end);
      }
      log_warn(
        "writer_append_wait: stall %d/%d, backing off", stalls, max_stalls);
      platform_sleep_ns(1000000LL << (stalls < 6 ? stalls : 6)); // 1ms..64ms
    } else {
      stalls = 0;
    }

    data = r.rest;
  }

  return writer_ok();
}

struct writer_result
writer_ok(void)
{
  struct writer_result r = { 0, { 0, 0 } };
  return r;
}

struct writer_result
writer_error(void)
{
  struct writer_result r = { 0, { 0, 0 } };
  r.error = 1;
  return r;
}

struct writer_result
writer_error_at(const void* beg, const void* end)
{
  struct writer_result r = { 0, { 0, 0 } };
  r.error = writer_error_fail;
  r.rest.beg = beg;
  r.rest.end = end;
  return r;
}

struct writer_result
writer_finished_at(const void* beg, const void* end)
{
  struct writer_result r = { 0, { 0, 0 } };
  r.error = writer_error_finished;
  r.rest.beg = beg;
  r.rest.end = end;
  return r;
}

size_t
shard_sink_pending_bytes(const struct shard_sink* s)
{
  return (s && s->pending_bytes) ? s->pending_bytes(s) : 0;
}

size_t
shard_pool_pending_bytes(const struct shard_pool* p)
{
  return (p && p->pending_bytes) ? p->pending_bytes(p) : 0;
}
