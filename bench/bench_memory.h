#pragma once

#include <stdint.h>

struct bench_memory
{
  uint64_t estimate_total_bytes;  // device bytes on GPU, heap bytes on CPU
  uint64_t estimate_pinned_bytes; // pinned host bytes on GPU, 0 on CPU
  uint64_t host_baseline_bytes;   // resident before the stream was created
  uint64_t host_peak_bytes;       // most resident memory held during the run
  int host_reading_failed;
  uint64_t device_used_bytes; // nonnegative observed device memory delta
  int device_overhead_valid;
  int64_t device_overhead_bytes; // observed device delta minus estimate
  // Device memory on GPU, the host difference on CPU. 0 if unavailable.
  uint64_t measured_bytes;
};

// Zero free-memory readings are unavailable.
void
bench_memory_record_device(struct bench_memory* mem,
                           uint64_t free_before,
                           uint64_t free_after);
