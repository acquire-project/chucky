#pragma once

#include <stdint.h>

// Sleep for the given number of nanoseconds.
void
platform_sleep_ns(int64_t ns);

// Return the OS page size in bytes.
size_t
platform_page_size(void);

// Allocate memory with the given alignment. Free with platform_aligned_free.
void*
platform_aligned_alloc(size_t alignment, size_t size);

void
platform_aligned_free(void* ptr);

// Monotonic clock for timing. Returns elapsed seconds since last call.
struct platform_clock
{
  int64_t last_ns;
};

float
platform_toc(struct platform_clock* clock);
