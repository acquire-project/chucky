#include "platform.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <malloc.h>

size_t
platform_page_size(void)
{
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return (size_t)si.dwPageSize;
}

void*
platform_aligned_alloc(size_t alignment, size_t size)
{
  return _aligned_malloc(size, alignment);
}

void
platform_aligned_free(void* ptr)
{
  _aligned_free(ptr);
}

void
platform_sleep_ns(int64_t ns)
{
  /* Sleep() has millisecond granularity; round up so we never sleep short. */
  DWORD ms = (DWORD)((ns + 999999LL) / 1000000LL);
  Sleep(ms);
}

static int64_t
monotonic_ns(void)
{
  static LARGE_INTEGER freq = {0};
  LARGE_INTEGER cnt;
  if (freq.QuadPart == 0)
    QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&cnt);
  /* Convert to nanoseconds: cnt * 1e9 / freq, avoiding overflow. */
  return (int64_t)(cnt.QuadPart / freq.QuadPart) * 1000000000LL
       + (int64_t)(cnt.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart;
}

float
platform_toc(struct platform_clock* clock)
{
  int64_t now = monotonic_ns();
  float elapsed = (now - clock->last_ns) / 1e9f;
  clock->last_ns = now;
  return elapsed;
}
