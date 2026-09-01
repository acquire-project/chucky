#pragma once

#include "gpu/host_batch.copy.h"
#include "lod/lod_plan.h"
#include "stream/types.aggregate.h"

#include <cuda.h>
#include <stdint.h>

struct shard_state;
struct compress_agg_array;

struct compress_agg_plan
{
  struct batch_aggregate_layout layout;
  uint32_t active_count_by_level[LOD_MAX_LEVELS];
};

struct flush_handoff
{
  struct aggregate_batch batch;
  CUevent compress_start;
  CUevent compress_end;
  CUevent aggregate_start;
  struct shard_state* shards_by_level[LOD_MAX_LEVELS];
};
