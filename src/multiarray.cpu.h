#pragma once

#include "types.stream.h"
#include "writer.h"

enum multiarray_writer_error
{
  multiarray_writer_ok = 0,
  multiarray_writer_fail = 1,
  multiarray_writer_finished = 2,
  multiarray_writer_not_flushable = 3,
};

struct multiarray_writer_result
{
  int error;
  struct slice rest;
};

struct multiarray_writer
{
  struct multiarray_writer_result (*update)(struct multiarray_writer* self,
                                            int array_index,
                                            struct slice data);
  struct multiarray_writer_result (*flush)(struct multiarray_writer* self);
};

struct multiarray_tile_stream_cpu;

struct multiarray_tile_stream_cpu*
multiarray_tile_stream_cpu_create(
  int n_arrays,
  const struct tile_stream_configuration configs[],
  struct shard_sink* sinks[]);

void
multiarray_tile_stream_cpu_destroy(struct multiarray_tile_stream_cpu* ms);

struct multiarray_writer*
multiarray_tile_stream_cpu_writer(struct multiarray_tile_stream_cpu* ms);
