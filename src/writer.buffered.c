#include "writer.buffered.h"

#include "platform/platform.h"

#include <stdlib.h>
#include <string.h>

struct buffered_slot
{
  size_t bytes;
  int64_t accepted_ns;
};

struct buffered_writer
{
  struct writer writer;
  struct writer* downstream;
  struct platform_mutex* mutex;
  struct platform_cond* changed;
  struct platform_thread* worker;
  unsigned char* data;
  struct buffered_slot* slots;
  size_t slot_bytes;
  size_t slot_count;
  size_t max_drain_slots;
  size_t capacity_bytes;
  size_t head;
  size_t tail;
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
    if (b->stats.occupied_slots) {
      // Snapshot a contiguous prefix of the backlog. Keep its slots occupied
      // until downstream releases the input; the producer uses all other slots.
      const int64_t first_accepted_ns = b->slots[b->head].accepted_ns;
      uint64_t acceptance_offsets_ns = 0;
      size_t bytes = 0, count = 0, next = b->head;
      while (count < b->stats.occupied_slots && count < b->max_drain_slots &&
             bytes < b->capacity_bytes - b->read_offset) {
        bytes += b->slots[next].bytes;
        acceptance_offsets_ns += b->slots[next].accepted_ns - first_accepted_ns;
        ++count;
        if (++next == b->slot_count)
          next = 0;
      }
      const unsigned char* beg = b->data + b->read_offset;
      platform_mutex_unlock(b->mutex);
      const int64_t start = platform_monotonic_ns();
      struct writer_result r =
        writer_append_wait(b->downstream, (struct slice){ beg, beg + bytes });
      const int64_t end = platform_monotonic_ns();
      size_t rest = remaining_bytes(&r, beg, beg + bytes);
      platform_mutex_lock(b->mutex);

      struct buffered_writer_stats* s = &b->stats;
      const uint64_t oldest_queue_ns = start - first_accepted_ns;
      s->queue_ns += count * oldest_queue_ns - acceptance_offsets_ns;
      if (oldest_queue_ns > s->max_queue_ns)
        s->max_queue_ns = oldest_queue_ns;
      record_time(&s->downstream_ns, &s->max_downstream_ns, end - start);
      if ((uint64_t)(end - first_accepted_ns) > s->max_completion_ns)
        s->max_completion_ns = end - first_accepted_ns;
      s->completed_slots += count;
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
        s->occupied_slots = 0;
      } else {
        s->occupied_slots -= count;
        b->head = next;
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
      b->stats.occupied_slots == b->slot_count) {
    const int64_t start = platform_monotonic_ns();
    do {
      platform_cond_wait(b->changed, b->mutex);
      status = append_status(b);
    } while (!status && b->stats.occupied_slots == b->slot_count);
    b->stats.backpressure_ns += platform_monotonic_ns() - start;
  }
  if (status || input.beg == input.end) {
    platform_mutex_unlock(b->mutex);
    return (struct writer_result){ status, input };
  }
  size_t bytes =
    (const unsigned char*)input.end - (const unsigned char*)input.beg;
  if (bytes > b->slot_bytes)
    bytes = b->slot_bytes;
  if (bytes > b->capacity_bytes - b->write_offset)
    bytes = b->capacity_bytes - b->write_offset;
  const size_t offset = b->write_offset;
  platform_mutex_unlock(b->mutex);

  // Every occupied slot holds at most slot_bytes, so a free slot guarantees
  // room for this copy. Stop at the ring boundary to keep each slot contiguous.
  // The worker cannot see this slot until publication, and never moves data.
  memcpy(b->data + offset, input.beg, bytes);

  platform_mutex_lock(b->mutex);
  status = append_status(b); // downstream may have terminated during the copy
  if (!status) {
    b->slots[b->tail] =
      (struct buffered_slot){ bytes, platform_monotonic_ns() };
    if (++b->tail == b->slot_count)
      b->tail = 0;
    b->write_offset += bytes;
    if (b->write_offset == b->capacity_bytes)
      b->write_offset = 0;
    b->stats.accepted_bytes += bytes;
    b->stats.pending_bytes += bytes;
    ++b->stats.occupied_slots;
    if (b->stats.pending_bytes > b->stats.peak_pending_bytes)
      b->stats.peak_pending_bytes = b->stats.pending_bytes;
    if (b->stats.occupied_slots > b->stats.peak_occupied_slots)
      b->stats.peak_occupied_slots = b->stats.occupied_slots;
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
  free(b->slots);
  free(b->data);
  free(b);
}

struct buffered_writer*
buffered_writer_create(struct writer* downstream,
                       const struct buffered_writer_config* config)
{
  if (!downstream || !downstream->append || !downstream->flush || !config ||
      !config->slot_bytes || !config->slot_count ||
      config->slot_count > SIZE_MAX / config->slot_bytes ||
      config->slot_count > SIZE_MAX / sizeof(struct buffered_slot) ||
      config->max_drain_slots > config->slot_count)
    return NULL;
  struct buffered_writer* b = calloc(1, sizeof(*b));
  if (!b)
    return NULL;
  b->writer =
    (struct writer){ buffered_append, buffered_flush, buffered_close };
  b->downstream = downstream;
  b->slot_bytes = config->slot_bytes;
  b->slot_count = config->slot_count;
  b->max_drain_slots =
    config->max_drain_slots ? config->max_drain_slots : config->slot_count;
  b->capacity_bytes = config->slot_count * b->slot_bytes;
  b->data = malloc(b->capacity_bytes);
  b->slots = calloc(config->slot_count, sizeof(*b->slots));
  b->mutex = platform_mutex_new();
  b->changed = platform_cond_new();
  if (!b->data || !b->slots || !b->mutex || !b->changed)
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
