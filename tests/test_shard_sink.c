#include "test_shard_sink.h"

#include <stdlib.h>
#include <string.h>

static int
test_sink_write(struct shard_writer* self,
                uint64_t offset,
                const void* beg,
                const void* end)
{
  struct test_shard_writer* w = (struct test_shard_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  if (offset + nbytes > w->capacity)
    return 1;
  memcpy(w->buf + offset, beg, nbytes);
  if (offset + nbytes > w->size)
    w->size = offset + nbytes;
  return 0;
}

static int
test_sink_finalize(struct shard_writer* self)
{
  struct test_shard_writer* w = (struct test_shard_writer*)self;
  w->finalized = 1;
  w->sink->finalize_count++;
  return 0;
}

static int
discard_write(struct shard_writer* self,
              uint64_t offset,
              const void* beg,
              const void* end)
{
  (void)self;
  (void)offset;
  (void)beg;
  (void)end;
  return 0;
}

static int
discard_finalize(struct shard_writer* self)
{
  (void)self;
  return 0;
}

static struct shard_writer*
test_sink_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  struct test_shard_sink* s = (struct test_shard_sink*)self;
  s->open_count++;

  if (level >= s->num_levels)
    return &s->discard_writer;
  if ((int)shard_index >= s->num_shards_per_level[level])
    return NULL;

  struct test_shard_writer* w = &s->writers[level][shard_index];
  if (!w->buf) {
    w->buf = (uint8_t*)calloc(1, s->per_shard_capacity);
    w->capacity = s->per_shard_capacity;
    w->base.write = test_sink_write;
    w->base.finalize = test_sink_finalize;
    w->sink = s;
  }
  w->level = level;
  w->shard_index = shard_index;
  w->finalized = 0;
  w->size = 0;
  return &w->base;
}

void
test_sink_init_multi(struct test_shard_sink* s,
                     int num_levels,
                     const int* num_shards_per_level,
                     size_t per_shard_capacity)
{
  memset(s, 0, sizeof(*s));
  s->base.open = test_sink_open;
  s->per_shard_capacity = per_shard_capacity;
  s->num_levels = num_levels;
  s->discard_writer.write = discard_write;
  s->discard_writer.finalize = discard_finalize;
  for (int lv = 0; lv < num_levels && lv < TEST_SHARD_SINK_MAX_LEVELS; ++lv)
    s->num_shards_per_level[lv] = num_shards_per_level[lv];
}

void
test_sink_init(struct test_shard_sink* s,
               int num_shards,
               size_t per_shard_capacity)
{
  int shards[1] = { num_shards };
  test_sink_init_multi(s, 1, shards, per_shard_capacity);
}

void
test_sink_free(struct test_shard_sink* s)
{
  for (int lv = 0; lv < TEST_SHARD_SINK_MAX_LEVELS; ++lv)
    for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i)
      free(s->writers[lv][i].buf);
  memset(s, 0, sizeof(*s));
}
