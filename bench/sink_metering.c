#include "sink_metering.h"

#include <stddef.h>

static int
metering_write(struct shard_writer* self,
               uint64_t offset,
               const void* beg,
               const void* end)
{
  struct metering_writer* w = (struct metering_writer*)self;
  platform_toc(&w->parent->clock);
  int rc = w->inner->write(w->inner, offset, beg, end);
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  w->parent->total_bytes += nbytes;
  accumulate_metric_ms(&w->parent->metric,
                       (float)(platform_toc(&w->parent->clock) * 1000.0),
                       nbytes,
                       0);
  return rc;
}

static int
metering_write_direct(struct shard_writer* self,
                      uint64_t offset,
                      const void* beg,
                      const void* end)
{
  struct metering_writer* w = (struct metering_writer*)self;
  platform_toc(&w->parent->clock);
  int rc = w->inner->write_direct(w->inner, offset, beg, end);
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  w->parent->total_bytes += nbytes;
  accumulate_metric_ms(&w->parent->metric,
                       (float)(platform_toc(&w->parent->clock) * 1000.0),
                       nbytes,
                       0);
  return rc;
}

static int
metering_truncate(struct shard_writer* self, uint64_t logical_size)
{
  struct metering_writer* w = (struct metering_writer*)self;
  return w->inner->truncate(w->inner, logical_size);
}

static int
metering_finalize(struct shard_writer* self)
{
  struct metering_writer* w = (struct metering_writer*)self;
  int rc = w->inner->finalize(w->inner);
  w->in_use = 0;
  w->inner = NULL;
  return rc;
}

static struct shard_writer*
metering_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  struct metering_sink* ms = (struct metering_sink*)self;
  struct shard_writer* inner = ms->inner->open(ms->inner, level, shard_index);
  if (!inner)
    return NULL;
  for (int i = 0; i < METER_MAX_WRITERS; ++i) {
    if (!ms->writers[i].in_use) {
      ms->writers[i].in_use = 1;
      ms->writers[i].inner = inner;
      ms->writers[i].base.write_direct =
        inner->write_direct ? metering_write_direct : NULL;
      ms->writers[i].base.truncate = inner->truncate ? metering_truncate : NULL;
      return &ms->writers[i].base;
    }
  }
  return NULL;
}

static struct io_event
metering_record_fence(struct shard_sink* self)
{
  struct metering_sink* ms = (struct metering_sink*)self;
  return ms->inner->record_fence(ms->inner);
}

static void
metering_wait_fence(struct shard_sink* self, struct io_event ev)
{
  struct metering_sink* ms = (struct metering_sink*)self;
  ms->inner->wait_fence(ms->inner, ev);
}

static int
metering_has_error(const struct shard_sink* self)
{
  const struct metering_sink* ms = (const struct metering_sink*)self;
  return ms->inner->has_error(ms->inner);
}

static size_t
metering_pending_bytes(const struct shard_sink* self)
{
  const struct metering_sink* ms = (const struct metering_sink*)self;
  return ms->inner->pending_bytes(ms->inner);
}

static size_t
metering_required_shard_alignment(const struct shard_sink* self)
{
  const struct metering_sink* ms = (const struct metering_sink*)self;
  return ms->inner->required_shard_alignment(ms->inner);
}

static int
metering_update_append(struct shard_sink* self,
                       uint8_t level,
                       uint8_t n_append,
                       const uint64_t* append_sizes)
{
  // Kept out of the sink metric: it pairs times with byte counts, and a
  // zero-byte sample becomes the best entry and reports the stage at no
  // bandwidth at all.
  struct metering_sink* ms = (struct metering_sink*)self;
  return ms->inner->update_append(ms->inner, level, n_append, append_sizes);
}

void
metering_sink_init(struct metering_sink* ms, struct shard_sink* inner)
{
  *ms = (struct metering_sink){
    .base = {
      .open = metering_open,
      .update_append = inner->update_append ? metering_update_append : NULL,
      .record_fence = inner->record_fence ? metering_record_fence : NULL,
      .wait_fence = inner->wait_fence ? metering_wait_fence : NULL,
      .has_error = inner->has_error ? metering_has_error : NULL,
      .pending_bytes = inner->pending_bytes ? metering_pending_bytes : NULL,
      .required_shard_alignment = inner->required_shard_alignment
        ? metering_required_shard_alignment : NULL,
    },
    .inner = inner,
    .metric = { .name = "Sink", .best_ms = 1e30f },
  };
  for (int i = 0; i < METER_MAX_WRITERS; ++i) {
    ms->writers[i] = (struct metering_writer){
      .base = { .write = metering_write, .finalize = metering_finalize },
      .parent = ms,
    };
  }
}
