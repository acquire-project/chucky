// Shared deterministic pyramid for filesystem and streaming S3 readback.
#pragma once

#include "dimension.h"
#include "ngff.h"
#include <stdint.h>
#include <string.h>

enum
{
  NGFF_READBACK_NT = 5,
  NGFF_READBACK_NY = 512,
  NGFF_READBACK_NX = 512,
  NGFF_READBACK_LEVELS = 3,
};

static const struct ngff_axis ngff_readback_axes[] = {
  { .type = ngff_axis_time, .unit = "second", .scale = 0.25 },
  { .type = ngff_axis_space, .unit = "micrometer", .scale = 0.5 },
  { .type = ngff_axis_space, .unit = "micrometer", .scale = 0.75 },
};

static inline void
ngff_readback_fill(void* output, int first_frame, int frames)
{
  uint8_t* bytes = output;
  for (int t = 0; t < frames; ++t)
    for (int y = 0; y < NGFF_READBACK_NY; ++y)
      for (int x = 0; x < NGFF_READBACK_NX; ++x) {
        // Positive affine ramp: spatial block means are exact integers.
        const uint16_t value =
          (uint16_t)(1 + 1024 * (first_frame + t) + 8 * y + 4 * x);
        const size_t i =
          ((size_t)t * NGFF_READBACK_NY + y) * NGFF_READBACK_NX + x;
        memcpy(bytes + i * sizeof(value), &value, sizeof(value));
      }
}

static inline void
ngff_readback_dimensions(struct dimension dims[3],
                         uint64_t frames,
                         uint64_t shard_frames,
                         uint64_t spatial_chunks)
{
  dims_create(
    dims, "tyx", (uint64_t[]){ frames, NGFF_READBACK_NY, NGFF_READBACK_NX });
  dims_set_chunk_sizes(dims, 3, (uint64_t[]){ 1, 128, 128 });
  dims[0].chunks_per_shard = shard_frames;
  dims[1].chunks_per_shard = dims[2].chunks_per_shard = spatial_chunks;
  dims[1].downsample = dims[2].downsample = 1;
}
