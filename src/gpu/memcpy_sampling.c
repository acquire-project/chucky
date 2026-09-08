#include "gpu/memcpy_sampling.h"

int
memcpy_timing_sample(uint64_t* small_copies)
{
  const uint64_t n = (*small_copies)++;
  return ((n ^ (n >> 6)) & 63u) == 0;
}
