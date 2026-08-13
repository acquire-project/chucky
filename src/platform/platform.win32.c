#include "platform/platform.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <malloc.h>
#include <psapi.h>

uint64_t
platform_process_id(void)
{
  return (uint64_t)GetCurrentProcessId();
}

size_t
platform_page_size(void)
{
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return (size_t)si.dwPageSize;
}

size_t
platform_page_alignment(void)
{
  size_t ps = platform_page_size();
  return ps > 0 ? ps : 4096;
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

size_t
platform_available_memory(void)
{
  MEMORYSTATUSEX memstat;
  memstat.dwLength = sizeof(memstat);
  if (GlobalMemoryStatusEx(&memstat) && memstat.ullAvailPhys > 0)
    return (size_t)memstat.ullAvailPhys;
  return 0;
}

uint64_t
platform_resident_memory(void)
{
  PROCESS_MEMORY_COUNTERS pmc;
  pmc.cb = sizeof(pmc);
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    return 0;
  return (uint64_t)pmc.WorkingSetSize;
}

uint64_t
platform_peak_resident_memory(void)
{
  PROCESS_MEMORY_COUNTERS pmc;
  pmc.cb = sizeof(pmc);
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    return 0;
  return (uint64_t)pmc.PeakWorkingSetSize;
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
  static LARGE_INTEGER freq = { 0 };
  LARGE_INTEGER cnt;
  if (freq.QuadPart == 0)
    QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&cnt);
  /* Convert to nanoseconds: cnt * 1e9 / freq, avoiding overflow. */
  return (int64_t)(cnt.QuadPart / freq.QuadPart) * 1000000000LL +
         (int64_t)(cnt.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart;
}

float
platform_toc(struct platform_clock* clock)
{
  int64_t now = monotonic_ns();
  float elapsed = (now - clock->last_ns) / 1e9f;
  clock->last_ns = now;
  return elapsed;
}

static BOOL CALLBACK
once_callback(PINIT_ONCE once, PVOID param, PVOID* ctx)
{
  (void)once;
  (void)ctx;
  void (*fn)(void) = (void (*)(void))param;
  fn();
  return TRUE;
}

void
platform_call_once(platform_once* flag, void (*fn)(void))
{
  InitOnceExecuteOnce(flag, once_callback, (PVOID)fn, NULL);
}

struct platform_thread
{
  HANDLE handle;
  void (*fn)(void*);
  void* arg;
};

static DWORD WINAPI
thread_trampoline(LPVOID p)
{
  struct platform_thread* t = (struct platform_thread*)p;
  t->fn(t->arg);
  return 0;
}

struct platform_thread*
platform_thread_start(void (*fn)(void*), void* arg)
{
  struct platform_thread* t =
    (struct platform_thread*)calloc(1, sizeof(struct platform_thread));
  if (!t)
    return NULL;
  t->fn = fn;
  t->arg = arg;
  t->handle = CreateThread(NULL, 0, thread_trampoline, t, 0, NULL);
  if (!t->handle) {
    free(t);
    return NULL;
  }
  return t;
}

int
platform_thread_join(struct platform_thread* t)
{
  if (!t)
    return -1;
  WaitForSingleObject(t->handle, INFINITE);
  CloseHandle(t->handle);
  free(t);
  return 0;
}

struct platform_mutex
{
  SRWLOCK lock;
};

struct platform_mutex*
platform_mutex_new(void)
{
  struct platform_mutex* m =
    (struct platform_mutex*)calloc(1, sizeof(struct platform_mutex));
  if (!m)
    return NULL;
  InitializeSRWLock(&m->lock);
  return m;
}

void
platform_mutex_free(struct platform_mutex* m)
{
  /* SRWLOCK has no destructor. */
  free(m);
}

void
platform_mutex_lock(struct platform_mutex* m)
{
  AcquireSRWLockExclusive(&m->lock);
}

void
platform_mutex_unlock(struct platform_mutex* m)
{
  ReleaseSRWLockExclusive(&m->lock);
}

struct platform_cond
{
  CONDITION_VARIABLE cv;
};

struct platform_cond*
platform_cond_new(void)
{
  struct platform_cond* c =
    (struct platform_cond*)calloc(1, sizeof(struct platform_cond));
  if (!c)
    return NULL;
  InitializeConditionVariable(&c->cv);
  return c;
}

void
platform_cond_free(struct platform_cond* c)
{
  /* CONDITION_VARIABLE has no destructor. */
  free(c);
}

void
platform_cond_wait(struct platform_cond* c, struct platform_mutex* m)
{
  SleepConditionVariableSRW(&c->cv, &m->lock, INFINITE, 0);
}

void
platform_cond_broadcast(struct platform_cond* c)
{
  WakeAllConditionVariable(&c->cv);
}

void
platform_cpu_pause(void)
{
#if defined(_M_IX86) || defined(_M_X64)
  YieldProcessor(); /* expands to _mm_pause() */
#elif defined(_M_ARM) || defined(_M_ARM64)
  YieldProcessor(); /* expands to __yield() */
#else
  SwitchToThread();
#endif
}

int
platform_default_thread_count(void)
{
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
}
