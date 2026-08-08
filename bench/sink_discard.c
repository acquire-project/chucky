#include "sink_discard.h"

#include <stddef.h>

static int
discard_shard_write(struct shard_writer* self,
                    uint64_t offset,
                    const void* beg,
                    const void* end)
{
  (void)offset;
  struct discard_shard_writer* w = (struct discard_shard_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  w->parent->total_bytes += nbytes;
  return 0;
}

static int
discard_shard_finalize(struct shard_writer* self)
{
  struct discard_shard_writer* w = (struct discard_shard_writer*)self;
  w->parent->shards_finalized++;
  return 0;
}

static struct shard_writer*
discard_shard_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
  (void)shard_index;
  struct discard_shard_sink* s = (struct discard_shard_sink*)self;
  return &s->writer.base;
}

// Fixed rather than the running platform's page size, so numbers taken on
// machines with different page sizes stay comparable.
#define DISCARD_SHARD_ALIGNMENT 4096

static size_t
discard_required_shard_alignment(const struct shard_sink* self)
{
  (void)self;
  return DISCARD_SHARD_ALIGNMENT;
}

void
discard_shard_sink_init(struct discard_shard_sink* s)
{
  *s = (struct discard_shard_sink){
    .base = { .open = discard_shard_open,
              .required_shard_alignment = discard_required_shard_alignment },
  };
  s->writer = (struct discard_shard_writer){
    .base = { .write = discard_shard_write,
              .finalize = discard_shard_finalize },
    .parent = s,
  };
}
