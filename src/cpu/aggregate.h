#pragma once

#include "stream/types.aggregate.h"

#include <stddef.h>

struct host_batch;
struct threadpool;

struct aggregate_cpu_workspace
{
  uint32_t* perm;
  size_t* permuted_sizes;
  size_t* offsets;
  size_t* chunk_sizes;
  void* data;
  size_t data_capacity;
};

struct aggregate_cpu_inputs
{
  const void* compressed_base;
  const size_t* comp_sizes_base;
  const uint32_t* gather;
  const struct batch_aggregate_layout* layout;
  struct aggregate_cpu_workspace* ws;
  struct threadpool* pool;
};

int
aggregate_cpu_batch_prepare_unified(const struct aggregate_cpu_inputs* in);

int
aggregate_cpu_batch_copy_to_host(const struct aggregate_cpu_inputs* in,
                                 const struct host_batch* host);
