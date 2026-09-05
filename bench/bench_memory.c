#include "bench_memory.h"

void
bench_memory_record_device(struct bench_memory* mem,
                           uint64_t free_before,
                           uint64_t free_after)
{
  mem->device_used_bytes = 0;
  mem->measured_bytes = 0;
  mem->device_overhead_valid = 0;
  mem->device_overhead_bytes = 0;
  if (!free_before || !free_after)
    return;
  if (free_before > free_after)
    mem->device_used_bytes = free_before - free_after;
  mem->measured_bytes = mem->device_used_bytes;
  if (!mem->estimate_total_bytes || mem->estimate_total_bytes > INT64_MAX ||
      free_before > INT64_MAX || free_after > INT64_MAX)
    return;
  const int64_t delta = (int64_t)free_before - (int64_t)free_after;
  const int64_t estimate = (int64_t)mem->estimate_total_bytes;
  if (delta < INT64_MIN + estimate)
    return;
  mem->device_overhead_bytes = delta - estimate;
  mem->device_overhead_valid = 1;
}
