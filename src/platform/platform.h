#pragma once

#include <stddef.h>
#include <stdint.h>

// Sleep for the given number of nanoseconds.
void
platform_sleep_ns(int64_t ns);

// Return the OS page size in bytes.
size_t
platform_page_size(void);

// Return the OS page size, or 4096 if the OS reports 0.
size_t
platform_page_alignment(void);

// Allocate memory with the given alignment. Free with platform_aligned_free.
void*
platform_aligned_alloc(size_t alignment, size_t size);

void
platform_aligned_free(void* ptr);

// Return the available physical memory in bytes, or 0 on failure.
size_t
platform_available_memory(void);

// Monotonic clock for timing. Returns elapsed seconds since last call.
struct platform_clock
{
  int64_t last_ns;
};

float
platform_toc(struct platform_clock* clock);

// One-time initialization. Thread-safe.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef INIT_ONCE platform_once;
#define PLATFORM_ONCE_INIT INIT_ONCE_STATIC_INIT
#else
#include <pthread.h>
typedef pthread_once_t platform_once;
#define PLATFORM_ONCE_INIT PTHREAD_ONCE_INIT
#endif

void
platform_call_once(platform_once* flag, void (*fn)(void));
