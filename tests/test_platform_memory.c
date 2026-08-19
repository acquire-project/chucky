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

static int
test_resident_memory_is_plausible(void)
{
  log_info("=== test_resident_memory_is_plausible ===");
  uint64_t resident = platform_resident_memory();
  CHECK(Fail, resident > 0); // 0 is how the reading reports failure
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
  uint64_t before = platform_resident_memory();
  char* block = (char*)malloc(BLOCK_BYTES);
  CHECK(Fail, block != NULL);
  touch_block(block);
  uint64_t during = platform_resident_memory();
  free(block);
  CHECK(Fail, during >= before + HALF_BLOCK);
  return 0;
Fail:
  return 1;
}

static int
test_peak_covers_memory_already_released(void)
{
  log_info("=== test_peak_covers_memory_already_released ===");
  uint64_t resident_before = platform_resident_memory();
  uint64_t peak_before = platform_peak_resident_memory();
  CHECK(Fail, peak_before > 0);

  char* block = (char*)malloc(BLOCK_BYTES);
  CHECK(Fail, block != NULL);
  touch_block(block);
  uint64_t resident_during = platform_resident_memory();
  uint64_t peak_during = platform_peak_resident_memory();
  free(block);

  uint64_t peak = platform_peak_resident_memory();
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
  failed += test_resident_memory_is_plausible();
  failed += test_resident_memory_follows_a_touched_block();
  failed += test_peak_covers_memory_already_released();
  if (failed)
    log_error("%d platform memory tests FAILED", failed);
  else
    log_info("all platform memory tests PASSED");
  return failed != 0;
}
