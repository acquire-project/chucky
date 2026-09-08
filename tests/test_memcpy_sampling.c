#include "gpu/memcpy_sampling.h"

#include <stdio.h>

#define REQUIRE(x)                                                             \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x);                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int
main(void)
{
  uint64_t phase = 0, positions = 0;
  for (unsigned block = 0; block < 128; ++block) {
    unsigned samples = 0;
    for (unsigned offset = 0; offset < 64; ++offset) {
      if (memcpy_timing_sample(&phase)) {
        REQUIRE(offset == block % 64);
        positions |= UINT64_C(1) << offset;
        samples++;
      }
    }
    REQUIRE(samples == 1);
  }
  REQUIRE(positions == UINT64_MAX);

  // Unsigned rollover preserves the policy, including the first sample.
  phase = UINT64_MAX - 63;
  for (unsigned i = 0; i < 64; ++i)
    REQUIRE(memcpy_timing_sample(&phase) == (i == 63));
  REQUIRE(phase == 0);
  REQUIRE(memcpy_timing_sample(&phase));

  uint64_t other = 0;
  REQUIRE(memcpy_timing_sample(&other));
  REQUIRE(!memcpy_timing_sample(&phase));
  REQUIRE(other == 1);
  return 0;
}
