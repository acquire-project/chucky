#include "test_platform.h"

#include "platform/platform.h"

#define POLL_STEP_MS 10

int
test_wait_flag(_Atomic int* flag, int timeout_ms)
{
  int waited_ms = 0;
  while (atomic_load(flag) == 0) {
    if (waited_ms >= timeout_ms)
      return -1;
    platform_sleep_ns((int64_t)POLL_STEP_MS * 1000000LL);
    waited_ms += POLL_STEP_MS;
  }
  return 0;
}
