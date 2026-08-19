#pragma once

#include <stddef.h>
#include <stdint.h>

// Sleep for the given number of nanoseconds.
void
platform_sleep_ns(int64_t ns);

uint64_t
platform_process_id(void);

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

// Physical memory this process holds right now, in bytes, or 0 on failure.
uint64_t
platform_resident_memory(void);

// Most physical memory this process has held at any point, in bytes, or 0 on
// failure. Never goes down, so it can be read once at the end of a run.
uint64_t
platform_peak_resident_memory(void);

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

// Threading primitives. All return NULL / nonzero on failure.
// Opaque types — implementations live in platform.{posix,darwin,win32}.c.
struct platform_thread;
struct platform_mutex;
struct platform_cond;

// Start a thread running fn(arg). Caller owns the returned handle and must
// platform_thread_join() it exactly once.
struct platform_thread*
platform_thread_start(void (*fn)(void*), void* arg);

// Join the thread, then free its handle.
int
platform_thread_join(struct platform_thread* t);

struct platform_mutex*
platform_mutex_new(void);

void
platform_mutex_free(struct platform_mutex* m);

void
platform_mutex_lock(struct platform_mutex* m);

void
platform_mutex_unlock(struct platform_mutex* m);

struct platform_cond*
platform_cond_new(void);

void
platform_cond_free(struct platform_cond* c);

// Atomically unlock m and wait for a signal/broadcast on c. Re-locks m before
// returning. Caller must hold m on entry.
void
platform_cond_wait(struct platform_cond* c, struct platform_mutex* m);

// Wake all waiters. Caller need not hold the associated mutex, but holding it
// during the broadcast avoids the lost-wake-up window for pools that toggle
// state under the lock.
void
platform_cond_broadcast(struct platform_cond* c);

// CPU-level pause/yield hint for short spin loops. PAUSE on x86, YIELD on
// AArch64, SwitchToThread on Windows fallback. Cheap; no syscall.
void
platform_cpu_pause(void);

// Number of online logical CPUs. Returns at least 1.
int
platform_default_thread_count(void);
