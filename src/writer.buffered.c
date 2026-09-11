#include "writer.buffered.h"

#include "platform/platform.h"

#include <stdlib.h>
#include <string.h>

struct buffered_writer
{
  struct writer writer;
  struct writer* downstream;
  struct platform_mutex* mutex;
  struct platform_cond* changed;
  struct platform_thread* worker;
  unsigned char* data;
  size_t capacity_bytes;
  size_t max_drain_bytes;
  size_t read_offset;
  size_t write_offset;
  struct buffered_writer_stats stats;
  int flush_requested;
  int flushed;
  int close_requested;
  int closed;
  int stop;
};

static void
record_time(uint64_t* sum, uint64_t* peak, uint64_t ns)
{
  *sum += ns;
  if (ns > *peak)
    *peak = ns;
}

// Only accept a suffix of the submitted batch, or an empty rest. Treat a broken
// downstream contract as failure without dereferencing its pointers.
static size_t
remaining_bytes(struct writer_result* r, const void* beg, const void* end)
{
  if (r->rest.beg == r->rest.end)
    return 0;
  if (r->rest.end == end && (uintptr_t)r->rest.beg >= (uintptr_t)beg &&
      (uintptr_t)r->rest.beg <= (uintptr_t)end)
    return (uintptr_t)end - (uintptr_t)r->rest.beg;
  r->error = writer_error_fail;
  return (uintptr_t)end - (uintptr_t)beg;
}

static void
worker_main(void* arg)
{
  struct buffered_writer* b = arg;
  platform_mutex_lock(b->mutex);
  for (;;) {
    if (b->stats.pending_bytes) {
      // Snapshot a contiguous byte prefix, independent of producer appends.
      // It remains pending until downstream releases the input.
      size_t bytes = b->stats.pending_bytes;
      if (bytes > b->max_drain_bytes)
        bytes = b->max_drain_bytes;
      if (bytes > b->capacity_bytes - b->read_offset)
        bytes = b->capacity_bytes - b->read_offset;
      const unsigned char* beg = b->data + b->read_offset;
      platform_mutex_unlock(b->mutex);
      const int64_t start = platform_monotonic_ns();
      struct writer_result r =
        writer_append_wait(b->downstream, (struct slice){ beg, beg + bytes });
      const int64_t end = platform_monotonic_ns();
      size_t rest = remaining_bytes(&r, beg, beg + bytes);
      platform_mutex_lock(b->mutex);

      struct buffered_writer_stats* s = &b->stats;
      record_time(&s->downstream_ns, &s->max_downstream_ns, end - start);
      ++s->completed_batches;
      if (bytes > s->max_batch_bytes)
        s->max_batch_bytes = bytes;
      s->forwarded_bytes += bytes - rest;
      s->pending_bytes -= bytes - rest;
      if (r.error || rest) {
        s->downstream_finished = r.error == writer_error_finished;
        s->failed = r.error != writer_error_finished || s->pending_bytes != 0;
        s->abandoned_bytes += s->pending_bytes;
        s->pending_bytes = 0;
      } else {
        b->read_offset += bytes;
        if (b->read_offset == b->capacity_bytes)
          b->read_offset = 0;
      }
    } else if (b->flush_requested && !b->flushed) {
      platform_mutex_unlock(b->mutex);
      struct writer_result r = writer_flush(b->downstream);
      platform_mutex_lock(b->mutex);
      b->stats.failed |= r.error != writer_error_ok;
      b->flushed = 1;
    } else if (b->close_requested && !b->closed) {
      platform_mutex_unlock(b->mutex);
      struct writer_result r = writer_close(b->downstream);
      platform_mutex_lock(b->mutex);
      b->stats.failed |= r.error != writer_error_ok;
      b->closed = 1;
    } else if (b->stop) {
      break;
    } else {
      platform_cond_wait(b->changed, b->mutex);
      continue;
    }
    platform_cond_broadcast(b->changed);
  }
  platform_mutex_unlock(b->mutex);
}

static int
append_status(const struct buffered_writer* b)
{
  if (b->stats.failed)
    return writer_error_fail;
  if (b->flush_requested || b->stats.downstream_finished)
    return writer_error_finished;
  return writer_error_ok;
}

static struct writer_result
buffered_append(struct writer* self, struct slice input)
{
  struct buffered_writer* b = (struct buffered_writer*)self;
  platform_mutex_lock(b->mutex);
  int status = append_status(b);
  if (!status && input.beg != input.end &&
      b->stats.pending_bytes == b->capacity_bytes) {
    const int64_t start = platform_monotonic_ns();
    do {
      platform_cond_wait(b->changed, b->mutex);
      status = append_status(b);
    } while (!status && b->stats.pending_bytes == b->capacity_bytes);
    b->stats.backpressure_ns += platform_monotonic_ns() - start;
  }
  if (status || input.beg == input.end) {
    platform_mutex_unlock(b->mutex);
    return (struct writer_result){ status, input };
  }
  size_t bytes =
    (const unsigned char*)input.end - (const unsigned char*)input.beg;
  if (bytes > b->capacity_bytes - b->stats.pending_bytes)
    bytes = b->capacity_bytes - b->stats.pending_bytes;
  if (bytes > b->capacity_bytes - b->write_offset)
    bytes = b->capacity_bytes - b->write_offset;
  const size_t offset = b->write_offset;
  platform_mutex_unlock(b->mutex);

  // The single producer owns the free suffix. The worker cannot see this copy
  // until publication and never moves data; active drains can run concurrently.
  memcpy(b->data + offset, input.beg, bytes);

  platform_mutex_lock(b->mutex);
  status = append_status(b); // downstream may have terminated during the copy
  if (!status) {
    b->write_offset += bytes;
    if (b->write_offset == b->capacity_bytes)
      b->write_offset = 0;
    b->stats.accepted_bytes += bytes;
    b->stats.pending_bytes += bytes;
    if (b->stats.pending_bytes > b->stats.peak_pending_bytes)
      b->stats.peak_pending_bytes = b->stats.pending_bytes;
    input.beg = (const unsigned char*)input.beg + bytes;
  }
  platform_cond_broadcast(b->changed);
  platform_mutex_unlock(b->mutex);
  return (struct writer_result){ status, input };
}

static struct writer_result
buffered_flush(struct writer* self)
{
  struct buffered_writer* b = (struct buffered_writer*)self;
  platform_mutex_lock(b->mutex);
  b->flush_requested = 1;
  platform_cond_broadcast(b->changed);
  while (!b->flushed)
    platform_cond_wait(b->changed, b->mutex);
  int failed = b->stats.failed;
  platform_mutex_unlock(b->mutex);
  return failed ? writer_error() : writer_ok();
}

static struct writer_result
buffered_close(struct writer* self)
{
  struct buffered_writer* b = (struct buffered_writer*)self;
  platform_mutex_lock(b->mutex);
  if (b->flush_requested) {
    b->close_requested = 1;
    platform_cond_broadcast(b->changed);
    while (!b->closed)
      platform_cond_wait(b->changed, b->mutex);
  }
  int failed = b->stats.failed;
  platform_mutex_unlock(b->mutex);
  return failed ? writer_error() : writer_ok();
}

static void
release(struct buffered_writer* b)
{
  platform_cond_free(b->changed);
  platform_mutex_free(b->mutex);
  free(b->data);
  free(b);
}

struct buffered_writer*
buffered_writer_create(struct writer* downstream,
                       const struct buffered_writer_config* config)
{
  if (!downstream || !downstream->append || !downstream->flush || !config ||
      !config->capacity_bytes ||
      config->max_drain_bytes > config->capacity_bytes)
    return NULL;
  struct buffered_writer* b = calloc(1, sizeof(*b));
  if (!b)
    return NULL;
  b->writer =
    (struct writer){ buffered_append, buffered_flush, buffered_close };
  b->downstream = downstream;
  b->capacity_bytes = config->capacity_bytes;
  b->max_drain_bytes =
    config->max_drain_bytes ? config->max_drain_bytes : config->capacity_bytes;
  b->data = malloc(b->capacity_bytes);
  b->mutex = platform_mutex_new();
  b->changed = platform_cond_new();
  if (!b->data || !b->mutex || !b->changed)
    goto fail;
  b->worker = platform_thread_start(worker_main, b);
  if (!b->worker)
    goto fail;
  return b;
fail:
  release(b);
  return NULL;
}

struct writer*
buffered_writer_as_writer(struct buffered_writer* b)
{
  return &b->writer;
}

struct buffered_writer_stats
buffered_writer_get_stats(struct buffered_writer* b)
{
  platform_mutex_lock(b->mutex);
  struct buffered_writer_stats stats = b->stats;
  platform_mutex_unlock(b->mutex);
  return stats;
}

int
buffered_writer_destroy(struct buffered_writer* b)
{
  if (!b)
    return 0;
  int failed = writer_flush(&b->writer).error != 0;
  failed |= writer_close(&b->writer).error != 0;
  platform_mutex_lock(b->mutex);
  b->stop = 1;
  platform_cond_broadcast(b->changed);
  platform_mutex_unlock(b->mutex);
  failed |= platform_thread_join(b->worker) != 0;
  release(b);
  return failed;
}
