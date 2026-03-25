#include "writer.h"

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
