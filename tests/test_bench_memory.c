#include "bench_report.h"

#include <stdint.h>
#include <stdio.h>

int
main(void)
{
  const struct
  {
    uint64_t before, after, estimate, used;
    int valid;
    int64_t overhead;
  } cases[] = {
    { 1000, 700, 200, 300, 1, 100 },
    { 1000, 700, 300, 300, 1, 0 },
    { 1000, 700, 400, 300, 1, -100 },
    { 1000, 1000, 200, 0, 1, -200 },
    { 700, 1000, 200, 0, 1, -500 },
    { 0, 700, 200, 0, 0, 0 },
    { 1000, 0, 200, 0, 0, 0 },
    { 1000, 700, 0, 300, 0, 0 },
    { 1000, 700, UINT64_MAX, 300, 0, 0 },
    { UINT64_MAX, 1, 200, UINT64_MAX - 1, 0, 0 },
    { 1, INT64_MAX, INT64_MAX, 0, 0, 0 },
  };
  struct bench_memory mem = { 0 };
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    mem.device_used_bytes = 123;
    mem.measured_bytes = 123;
    mem.device_overhead_valid = 1;
    mem.device_overhead_bytes = 123;
    mem.estimate_total_bytes = cases[i].estimate;
    bench_memory_record_device(&mem, cases[i].before, cases[i].after);
    if (mem.device_used_bytes != cases[i].used ||
        mem.measured_bytes != cases[i].used ||
        mem.device_overhead_valid != cases[i].valid ||
        mem.device_overhead_bytes != cases[i].overhead) {
      fprintf(stderr, "Memory reading case %zu failed\n", i);
      return 1;
    }
  }
  return 0;
}
