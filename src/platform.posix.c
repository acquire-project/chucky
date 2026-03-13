#include "platform.h"

#include <stdlib.h>
#include <time.h>
#include <unistd.h>

size_t
platform_page_size(void)
{
  return (size_t)sysconf(_SC_PAGESIZE);
}

void*
platform_aligned_alloc(size_t alignment, size_t size)
{
  return aligned_alloc(alignment, size);
}

void
platform_aligned_free(void* ptr)
{
  free(ptr);
}

void
platform_sleep_ns(int64_t ns)
{
  struct timespec ts = {
    .tv_sec = ns / 1000000000LL,
    .tv_nsec = ns % 1000000000LL,
  };
  nanosleep(&ts, NULL);
}

static int64_t
monotonic_ns(void)
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (int64_t)now.tv_sec * 1000000000LL + now.tv_nsec;
}

float
platform_toc(struct platform_clock* clock)
{
  int64_t now = monotonic_ns();
  float elapsed = (now - clock->last_ns) / 1e9f;
  clock->last_ns = now;
  return elapsed;
}
