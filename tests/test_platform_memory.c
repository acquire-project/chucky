#include "chucky_log.h"
#include "platform/platform.h"
#include "util/prelude.h"

#include <stdlib.h>

#define BLOCK_BYTES (64u << 20)

// Anything short of the whole block would still catch a kilobyte-reported-as-
// bytes mix-up by three orders of magnitude, so half is a loose enough floor.
#define HALF_BLOCK (BLOCK_BYTES / 2)

// One write per page, so every page is really resident. The writes go through
// a volatile pointer because a plain memset here is dead code: clang drops it
// and the allocation with it.
static void
touch_block(char* block)
{
  volatile char* pages = (volatile char*)block;
  const size_t page = platform_page_alignment();
  for (size_t i = 0; i < BLOCK_BYTES; i += page)
    pages[i] = 1;
}

// Linux batches the page counters behind /proc/self/statm, so a process a few
// tens of milliseconds old reads nothing resident however many pages it has
// touched. Only elapsed time clears it. The two tests that compare readings
// need the counter running before they start.
static void
wait_for_the_page_counter(void)
{
  for (int i = 0; i < 200; ++i) {
    uint64_t bytes = 0;
    if (platform_resident_memory(&bytes) != 0 || bytes > 0)
      return;
    platform_sleep_ns(1000000);
  }
  log_warn("resident memory still reads 0 after 200 ms");
}

static int
test_resident_memory_is_plausible(void)
{
  log_info("=== test_resident_memory_is_plausible ===");
  uint64_t resident = 0;
  CHECK(Fail, platform_resident_memory(&resident) == 0);
  CHECK(Fail, resident < ((uint64_t)1 << 40));
  return 0;
Fail:
  return 1;
}

static int
test_resident_memory_follows_a_touched_block(void)
{
  // Must run before any other test here allocates a block: an allocator that
  // keeps the freed pages leaves the next baseline already inflated.
  log_info("=== test_resident_memory_follows_a_touched_block ===");
  uint64_t before = 0, during = 0;
  int reads_failed = platform_resident_memory(&before);
  char* block = (char*)malloc(BLOCK_BYTES);
  CHECK(Fail, block != NULL);
  touch_block(block);
  reads_failed |= platform_resident_memory(&during);
  free(block);
  CHECK(Fail, reads_failed == 0);
  CHECK(Fail, during >= before + HALF_BLOCK);
  return 0;
Fail:
  return 1;
}

static int
test_peak_covers_memory_already_released(void)
{
  log_info("=== test_peak_covers_memory_already_released ===");
  uint64_t resident_before = 0, peak_before = 0;
  int reads_failed = platform_resident_memory(&resident_before);
  reads_failed |= platform_peak_resident_memory(&peak_before);

  char* block = (char*)malloc(BLOCK_BYTES);
  CHECK(Fail, block != NULL);
  touch_block(block);
  uint64_t resident_during = 0, peak_during = 0;
  reads_failed |= platform_resident_memory(&resident_during);
  reads_failed |= platform_peak_resident_memory(&peak_during);
  free(block);

  uint64_t peak = 0;
  reads_failed |= platform_peak_resident_memory(&peak);
  CHECK(Fail, reads_failed == 0);
  log_info("  resident %llu -> %llu, peak %llu -> %llu -> %llu, block %u",
           (unsigned long long)resident_before,
           (unsigned long long)resident_during,
           (unsigned long long)peak_before,
           (unsigned long long)peak_during,
           (unsigned long long)peak,
           BLOCK_BYTES);
  // Each reading is checked against the block rather than against the other
  // one: they come from different accounting in the kernel, and the peak can
  // trail the resident reading by a few pages.
  CHECK(Fail, resident_during >= HALF_BLOCK);
  CHECK(Fail, peak >= HALF_BLOCK);
  CHECK(Fail, peak >= peak_before);
  return 0;
Fail:
  return 1;
}

int
main(void)
{
  int failed = 0;
  wait_for_the_page_counter();
  failed += test_resident_memory_is_plausible();
  failed += test_resident_memory_follows_a_touched_block();
  failed += test_peak_covers_memory_already_released();
  if (failed)
    log_error("%d platform memory tests FAILED", failed);
  else
    log_info("all platform memory tests PASSED");
  return failed != 0;
}
