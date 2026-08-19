#define _GNU_SOURCE

#include "chucky_log.h"
#include "platform/platform.h"
#include "util/prelude.h"

#ifdef __linux__
#include <sched.h>
#endif

static int
test_thread_count_is_at_least_one(void)
{
  log_info("=== test_thread_count_is_at_least_one ===");
  int n = platform_default_thread_count();
  log_info("  %d threads", n);
  CHECK(Fail, n >= 1);
  return 0;
Fail:
  return 1;
}

#ifdef __linux__
// A batch scheduler hands out part of a machine the same way, so this is the
// case that decides how many threads a benchmark starts on a cluster.
static int
test_thread_count_follows_the_allowed_cores(void)
{
  log_info("=== test_thread_count_follows_the_allowed_cores ===");
  cpu_set_t original;
  if (sched_getaffinity(0, sizeof(original), &original) != 0) {
    log_info("  no affinity mask available, skipping");
    return 0;
  }
  if (CPU_COUNT(&original) < 2) {
    log_info("  only one core allowed, nothing to narrow");
    return 0;
  }

  int first = -1;
  for (int cpu = 0; cpu < CPU_SETSIZE && first < 0; ++cpu)
    if (CPU_ISSET(cpu, &original))
      first = cpu;

  cpu_set_t one;
  CPU_ZERO(&one);
  CPU_SET(first, &one);
  CHECK(Restore, sched_setaffinity(0, sizeof(one), &one) == 0);

  int narrowed = platform_default_thread_count();
  CHECK(Restore, sched_setaffinity(0, sizeof(original), &original) == 0);
  log_info("  %d allowed -> %d threads", CPU_COUNT(&original), narrowed);
  CHECK(Fail, narrowed == 1);
  return 0;
Restore:
  sched_setaffinity(0, sizeof(original), &original);
Fail:
  return 1;
}
#endif

int
main(void)
{
  int failed = 0;
  failed += test_thread_count_is_at_least_one();
#ifdef __linux__
  failed += test_thread_count_follows_the_allowed_cores();
#endif
  if (failed)
    log_error("%d platform thread tests FAILED", failed);
  else
    log_info("all platform thread tests PASSED");
  return failed != 0;
}
