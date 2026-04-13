#pragma once

#include "platform/platform.h"
#include "util/metric.h"
#include "writer.h"

#include <stddef.h>

#define METER_MAX_WRITERS 32

struct metering_writer
{
  struct shard_writer base;
  struct shard_writer* inner;
  struct metering_sink* parent;
  int in_use;
};

struct metering_sink
{
  struct shard_sink base;
  struct shard_sink* inner;
  struct metering_writer writers[METER_MAX_WRITERS];
  size_t total_bytes;
  struct stream_metric metric;
  struct platform_clock clock;
};

void
metering_sink_init(struct metering_sink* ms, struct shard_sink* inner);
