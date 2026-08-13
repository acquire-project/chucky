#include "chucky_log.h"
#include "platform/platform.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

#define BLOCK_BYTES (64u << 20)

// Anything short of the whole block would still catch a kilobyte-reported-as-
// bytes mix-up by three orders of magnitude, so half is a loose enough floor.
#define GROWTH_FLOOR (BLOCK_BYTES / 2)

static volatile char sink;

// Touch every page and read one back, so the pages are really resident and the
// compiler cannot drop the writes.
static void
touch_block(char* block)
{
  memset(block, 1, BLOCK_BYTES);
  sink = block[BLOCK_BYTES - 1];
}

static int
test_resident_memory_is_plausible(void)
{
  log_info("=== test_resident_memory_is_plausible ===");
  uint64_t resident = platform_resident_memory();
  CHECK(Fail, resident > (1u << 20));
  CHECK(Fail, resident < ((uint64_t)1 << 40));
  return 0;
Fail:
  return 1;
}

static int
test_resident_memory_follows_a_touched_block(void)
{
  log_info("=== test_resident_memory_follows_a_touched_block ===");
  uint64_t before = platform_resident_memory();
  char* block = (char*)malloc(BLOCK_BYTES);
  CHECK(Fail, block != NULL);
  touch_block(block);
  uint64_t during = platform_resident_memory();
  free(block);
  CHECK(Fail, during >= before + GROWTH_FLOOR);
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
  free(block);

  uint64_t peak = platform_peak_resident_memory();
  CHECK(Fail, peak >= peak_before);
  CHECK(Fail, peak >= resident_before + GROWTH_FLOOR);
  CHECK(Fail, peak >= platform_resident_memory());
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
