#pragma once

#include "defs.limits.h"
#include "dimension.h"
#include "types.codec.h"
#include "types.lod.h"

#include <stddef.h>
#include <stdint.h>

struct stream_metric
{
  const char* name;
  float ms;            // cumulative
  float best_ms;       // best single measurement (1e30f = not yet measured)
  double input_bytes;  // cumulative bytes read by stage
  double output_bytes; // cumulative bytes written by stage
  int count;
};

struct stream_metrics
{
  struct stream_metric memcpy;
  struct stream_metric h2d;
  struct stream_metric lod_gather;
  struct stream_metric lod_reduce;
  struct stream_metric lod_append_fold;
  struct stream_metric lod_morton_chunk;
  struct stream_metric scatter;
  struct stream_metric compress;
  struct stream_metric aggregate;
  struct stream_metric d2h;
  struct stream_metric sink;
};

struct tile_stream_configuration
{
  size_t buffer_capacity_bytes;
  enum dtype dtype;
  uint8_t rank;
  struct dimension* dimensions;
  struct codec_config codec;
  enum lod_reduce_method reduce_method;
  enum lod_reduce_method append_reduce_method;
  int max_nlod; // <0 = auto, 0 = none, N>0 = N downsampled levels
  uint8_t epochs_per_batch; // K: 0 = auto (target_batch_chunks), must be pow2
  uint32_t
    target_batch_chunks; // minimum chunks per compress batch (default 1024)
  float metadata_update_interval_s;
  size_t
    shard_alignment; // 0 = no padding; platform_page_size() for unbuffered IO
};

struct tile_stream_status
{
  int nlod;
  int append_downsample;
  uint32_t epochs_per_batch;
  size_t max_compressed_size;
  enum dtype dtype;
  struct codec_config codec;
  size_t codec_batch_size;
  uint32_t batch_accumulated;
  int pool_current;
  int flush_pending;
};
