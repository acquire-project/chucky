#include "platform/platform.h"

#include <mach/mach.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

size_t
platform_page_size(void)
{
  return (size_t)sysconf(_SC_PAGESIZE);
}

size_t
platform_page_alignment(void)
{
  size_t ps = platform_page_size();
  return ps > 0 ? ps : 4096;
}

size_t
platform_available_memory(void)
{
  mach_port_t host = mach_host_self();
  vm_statistics64_data_t vm;
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count) !=
      KERN_SUCCESS)
    return 0;
  return ((size_t)vm.free_count + (size_t)vm.inactive_count) *
         platform_page_size();
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

void
platform_call_once(platform_once* flag, void (*fn)(void))
{
  pthread_once(flag, fn);
}
