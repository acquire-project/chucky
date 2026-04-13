#pragma once

#include "writer.h"

#include <stddef.h>

struct discard_shard_writer
{
  struct shard_writer base;
  struct discard_shard_sink* parent;
};

struct discard_shard_sink
{
  struct shard_sink base;
  struct discard_shard_writer writer;
  size_t total_bytes;
  size_t shards_finalized;
};

void
discard_shard_sink_init(struct discard_shard_sink* s);
