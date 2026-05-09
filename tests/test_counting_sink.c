#include "test_counting_sink.h"

#include <stddef.h>

static int
counting_write(struct shard_writer* self,
               uint64_t offset,
               const void* beg,
               const void* end)
{
  struct counting_writer* w = (struct counting_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  atomic_fetch_add(&w->parent->copy_calls, 1);
  atomic_fetch_add(&w->parent->copy_bytes, (uint64_t)nbytes);
  return w->inner->write(w->inner, offset, beg, end);
}

static int
counting_write_direct(struct shard_writer* self,
                      uint64_t offset,
                      const void* beg,
                      const void* end)
{
  struct counting_writer* w = (struct counting_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  atomic_fetch_add(&w->parent->direct_calls, 1);
  atomic_fetch_add(&w->parent->direct_bytes, (uint64_t)nbytes);
  return w->inner->write_direct(w->inner, offset, beg, end);
}

static int
counting_truncate(struct shard_writer* self, uint64_t logical_size)
{
  struct counting_writer* w = (struct counting_writer*)self;
  return w->inner->truncate(w->inner, logical_size);
}

static int
counting_finalize(struct shard_writer* self)
{
  struct counting_writer* w = (struct counting_writer*)self;
  int rc = w->inner->finalize(w->inner);
  w->in_use = 0;
  w->inner = NULL;
  return rc;
}

static struct shard_writer*
counting_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  struct counting_sink* cs = (struct counting_sink*)self;
  struct shard_writer* inner = cs->inner->open(cs->inner, level, shard_index);
  if (!inner)
    return NULL;
  for (int i = 0; i < COUNTING_SINK_MAX_WRITERS; ++i) {
    if (!cs->writers[i].in_use) {
      cs->writers[i].in_use = 1;
      cs->writers[i].inner = inner;
      cs->writers[i].base.write_direct =
        inner->write_direct ? counting_write_direct : NULL;
      cs->writers[i].base.truncate = inner->truncate ? counting_truncate : NULL;
      return &cs->writers[i].base;
    }
  }
  return NULL;
}

static int
counting_update_append(struct shard_sink* self,
                       uint8_t level,
                       uint8_t n_append,
                       const uint64_t* append_sizes)
{
  struct counting_sink* cs = (struct counting_sink*)self;
  return cs->inner->update_append(cs->inner, level, n_append, append_sizes);
}

static struct io_event
counting_record_fence(struct shard_sink* self)
{
  struct counting_sink* cs = (struct counting_sink*)self;
  return cs->inner->record_fence(cs->inner);
}

static void
counting_wait_fence(struct shard_sink* self, struct io_event ev)
{
  struct counting_sink* cs = (struct counting_sink*)self;
  cs->inner->wait_fence(cs->inner, ev);
}

static int
counting_has_error(const struct shard_sink* self)
{
  const struct counting_sink* cs = (const struct counting_sink*)self;
  return cs->inner->has_error(cs->inner);
}

static size_t
counting_pending_bytes(const struct shard_sink* self)
{
  const struct counting_sink* cs = (const struct counting_sink*)self;
  return cs->inner->pending_bytes(cs->inner);
}

static size_t
counting_required_shard_alignment(const struct shard_sink* self)
{
  const struct counting_sink* cs = (const struct counting_sink*)self;
  return cs->inner->required_shard_alignment(cs->inner);
}

void
counting_sink_init(struct counting_sink* cs, struct shard_sink* inner)
{
  *cs = (struct counting_sink){
    .base = {
      .open = counting_open,
      .update_append = inner->update_append ? counting_update_append : NULL,
      .record_fence = inner->record_fence ? counting_record_fence : NULL,
      .wait_fence = inner->wait_fence ? counting_wait_fence : NULL,
      .has_error = inner->has_error ? counting_has_error : NULL,
      .pending_bytes = inner->pending_bytes ? counting_pending_bytes : NULL,
      .required_shard_alignment = inner->required_shard_alignment
        ? counting_required_shard_alignment : NULL,
    },
    .inner = inner,
  };
  for (int i = 0; i < COUNTING_SINK_MAX_WRITERS; ++i) {
    cs->writers[i] = (struct counting_writer){
      .base = { .write = counting_write, .finalize = counting_finalize },
      .parent = cs,
    };
  }
}

void
counting_sink_path_counts(const struct counting_sink* cs,
                          uint64_t* out_copy_calls,
                          uint64_t* out_direct_calls,
                          uint64_t* out_copy_bytes,
                          uint64_t* out_direct_bytes)
{
  if (out_copy_calls)
    *out_copy_calls = atomic_load(&cs->copy_calls);
  if (out_direct_calls)
    *out_direct_calls = atomic_load(&cs->direct_calls);
  if (out_copy_bytes)
    *out_copy_bytes = atomic_load(&cs->copy_bytes);
  if (out_direct_bytes)
    *out_direct_bytes = atomic_load(&cs->direct_bytes);
}
