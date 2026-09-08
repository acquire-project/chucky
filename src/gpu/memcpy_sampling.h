#pragma once

#include <stdint.h>

// A provisional instrumentation policy, not a copy-kernel crossover.
#define MEMCPY_TIMING_FULL_BYTES (8u << 10)

// One of each 64 small copies. Rotate the sampled position between blocks so
// power-of-two page/staging boundaries do not always select the same offset.
// The first copy is measured, including streams shorter than one block.
static inline int
memcpy_timing_sample(uint64_t* small_copies)
{
  const uint64_t n = (*small_copies)++;
  return ((n ^ (n >> 6)) & 63u) == 0;
}
