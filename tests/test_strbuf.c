#include "chucky_log.h"
#include "util/prelude.h"
#include "util/strbuf.h"

#include <stdint.h>
#include <string.h>

static int
test_zero_init_is_usable(void)
{
  log_info("=== test_zero_init_is_usable ===");
  struct strbuf sb = { 0 };
  CHECK(Fail, strbuf_len(&sb) == 0);
  CHECK(Fail, strcmp(strbuf_cstr(&sb), "") == 0);
  strbuf_free(&sb); // no-op on zero-init
  return 0;
Fail:
  return 1;
}

static int
test_inline_append(void)
{
  log_info("=== test_inline_append ===");
  struct strbuf sb = { 0 };
  CHECK(Fail, strbuf_append_cstr(&sb, "hello") == 0);
  CHECK(Fail, strbuf_append(&sb, ", ", 2) == 0);
  CHECK(Fail, strbuf_append_cstr(&sb, "world") == 0);
  CHECK(Fail, strbuf_len(&sb) == 12);
  CHECK(Fail, strcmp(strbuf_cstr(&sb), "hello, world") == 0);
  // Content must fit in inline storage.
  CHECK(Fail, sb.beg == sb.inline_buf);
  strbuf_free(&sb);
  return 0;
Fail:
  strbuf_free(&sb);
  return 1;
}

static int
test_spill_to_heap(void)
{
  log_info("=== test_spill_to_heap ===");
  struct strbuf sb = { 0 };
  // Write something larger than STRBUF_INLINE_CAP (128) to force heap spill.
  char big[512];
  memset(big, 'x', sizeof(big));
  CHECK(Fail, strbuf_append(&sb, big, sizeof(big)) == 0);
  CHECK(Fail, strbuf_len(&sb) == sizeof(big));
  // Must have moved off the inline buffer.
  CHECK(Fail, sb.beg != sb.inline_buf);
  // Content preserved.
  CHECK(Fail, memcmp(sb.beg, big, sizeof(big)) == 0);
  // NUL terminator at end.
  CHECK(Fail, sb.beg[sizeof(big)] == 0);
  strbuf_free(&sb);
  return 0;
Fail:
  strbuf_free(&sb);
  return 1;
}

static int
test_appendf(void)
{
  log_info("=== test_appendf ===");
  struct strbuf sb = { 0 };
  CHECK(Fail, strbuf_appendf(&sb, "%s/%d/%s", "pfx", 42, "suffix") == 0);
  CHECK(Fail, strcmp(strbuf_cstr(&sb), "pfx/42/suffix") == 0);
  // Append again without reset.
  CHECK(Fail, strbuf_appendf(&sb, "/more%d", 7) == 0);
  CHECK(Fail, strcmp(strbuf_cstr(&sb), "pfx/42/suffix/more7") == 0);
  strbuf_free(&sb);
  return 0;
Fail:
  strbuf_free(&sb);
  return 1;
}

static int
test_appendf_grows_past_inline(void)
{
  log_info("=== test_appendf_grows_past_inline ===");
  struct strbuf sb = { 0 };
  // vsnprintf's first attempt will run out of inline space; appendf should
  // grow and retry.
  for (int i = 0; i < 50; ++i)
    CHECK(Fail, strbuf_appendf(&sb, "segment-%02d/", i) == 0);
  // 50 * 11 = 550 bytes. Definitely past inline.
  CHECK(Fail, strbuf_len(&sb) == 50 * 11);
  CHECK(Fail, sb.beg != sb.inline_buf);
  strbuf_free(&sb);
  return 0;
Fail:
  strbuf_free(&sb);
  return 1;
}

static int
test_reset_keeps_capacity(void)
{
  log_info("=== test_reset_keeps_capacity ===");
  struct strbuf sb = { 0 };
  char big[512];
  memset(big, 'y', sizeof(big));
  CHECK(Fail, strbuf_append(&sb, big, sizeof(big)) == 0);
  char* heap_ptr = sb.beg;
  size_t heap_cap = (size_t)(sb.cap_end - sb.beg);
  CHECK(Fail, heap_ptr != sb.inline_buf);

  strbuf_reset(&sb);
  CHECK(Fail, strbuf_len(&sb) == 0);
  // Same allocation still in use.
  CHECK(Fail, sb.beg == heap_ptr);
  CHECK(Fail, (size_t)(sb.cap_end - sb.beg) == heap_cap);
  // Re-fill.
  CHECK(Fail, strbuf_append_cstr(&sb, "short") == 0);
  CHECK(Fail, sb.beg == heap_ptr); // no realloc needed
  strbuf_free(&sb);
  return 0;
Fail:
  strbuf_free(&sb);
  return 1;
}

static int
test_set(void)
{
  log_info("=== test_set ===");
  struct strbuf sb = { 0 };
  CHECK(Fail, strbuf_set(&sb, "first") == 0);
  CHECK(Fail, strcmp(strbuf_cstr(&sb), "first") == 0);
  CHECK(Fail, strbuf_set(&sb, "second (longer)") == 0);
  CHECK(Fail, strcmp(strbuf_cstr(&sb), "second (longer)") == 0);
  strbuf_free(&sb);
  return 0;
Fail:
  strbuf_free(&sb);
  return 1;
}

// Set on a heap-backed buffer should reuse the existing allocation if it fits.
static int
test_set_reuses_heap_allocation(void)
{
  log_info("=== test_set_reuses_heap_allocation ===");
  struct strbuf sb = { 0 };
  char big[512];
  memset(big, 'z', sizeof(big));
  CHECK(Fail, strbuf_append(&sb, big, sizeof(big)) == 0);
  char* heap_ptr = sb.beg;
  CHECK(Fail, heap_ptr != sb.inline_buf);

  CHECK(Fail, strbuf_set(&sb, "short") == 0);
  // Same heap allocation must be reused (no free+malloc cycle).
  CHECK(Fail, sb.beg == heap_ptr);
  CHECK(Fail, strbuf_len(&sb) == 5);
  CHECK(Fail, strcmp(strbuf_cstr(&sb), "short") == 0);
  strbuf_free(&sb);
  return 0;
Fail:
  strbuf_free(&sb);
  return 1;
}

// Reset on a zero-init buffer should be a no-op (no crash).
static int
test_reset_on_zero_init(void)
{
  log_info("=== test_reset_on_zero_init ===");
  struct strbuf sb = { 0 };
  strbuf_reset(&sb);
  CHECK(Fail, strbuf_len(&sb) == 0);
  CHECK(Fail, strcmp(strbuf_cstr(&sb), "") == 0);
  CHECK(Fail, sb.beg == NULL); // still zero-init
  strbuf_free(&sb);
  return 0;
Fail:
  return 1;
}

// Reserve(0) on a zero-init buffer should activate inline storage without
// touching the heap.
static int
test_reserve_zero(void)
{
  log_info("=== test_reserve_zero ===");
  struct strbuf sb = { 0 };
  CHECK(Fail, strbuf_reserve(&sb, 0) == 0);
  CHECK(Fail, sb.beg == sb.inline_buf);
  CHECK(Fail, strbuf_len(&sb) == 0);
  CHECK(Fail, strcmp(strbuf_cstr(&sb), "") == 0);
  strbuf_free(&sb);
  return 0;
Fail:
  strbuf_free(&sb);
  return 1;
}

// Exercise the boundary between the first vsnprintf fitting and triggering
// the grow+retry path: fill inline to within 1 byte of its capacity, then
// appendf one more character. Exits the inline buffer by exactly one byte.
static int
test_appendf_at_inline_boundary(void)
{
  log_info("=== test_appendf_at_inline_boundary ===");
  struct strbuf sb = { 0 };
  // Append STRBUF_INLINE_CAP - 1 bytes — fits exactly (leaving NUL slot).
  char fill[STRBUF_INLINE_CAP - 1];
  memset(fill, 'x', sizeof(fill));
  CHECK(Fail, strbuf_append(&sb, fill, sizeof(fill)) == 0);
  CHECK(Fail, sb.beg == sb.inline_buf);
  CHECK(Fail, strbuf_len(&sb) == STRBUF_INLINE_CAP - 1);

  // One more byte would overflow inline (no room for content + NUL); must
  // grow and retry.
  CHECK(Fail, strbuf_appendf(&sb, "%c", '!') == 0);
  CHECK(Fail, sb.beg != sb.inline_buf); // moved to heap
  CHECK(Fail, strbuf_len(&sb) == STRBUF_INLINE_CAP);
  CHECK(Fail, sb.beg[STRBUF_INLINE_CAP - 1] == '!');
  CHECK(Fail, sb.beg[STRBUF_INLINE_CAP] == 0);
  strbuf_free(&sb);
  return 0;
Fail:
  strbuf_free(&sb);
  return 1;
}

// Oversized reserve request must fail via the overflow guard, not wrap to
// an absurdly small allocation and write past it.
static int
test_reserve_overflow_guard(void)
{
  log_info("=== test_reserve_overflow_guard ===");
  struct strbuf sb = { 0 };
  CHECK(Fail, strbuf_reserve(&sb, SIZE_MAX) != 0);
  // Buffer should remain usable after the failed reserve.
  CHECK(Fail, strbuf_append_cstr(&sb, "ok") == 0);
  CHECK(Fail, strcmp(strbuf_cstr(&sb), "ok") == 0);
  strbuf_free(&sb);
  return 0;
Fail:
  strbuf_free(&sb);
  return 1;
}

// Appending raw bytes with embedded NULs preserves the bytes; the public
// C-string view ends at the first NUL but the length is correct.
static int
test_binary_with_embedded_nul(void)
{
  log_info("=== test_binary_with_embedded_nul ===");
  struct strbuf sb = { 0 };
  const char bytes[] = { 'a', 0, 'b', 'c' };
  CHECK(Fail, strbuf_append(&sb, bytes, sizeof(bytes)) == 0);
  CHECK(Fail, strbuf_len(&sb) == sizeof(bytes));
  CHECK(Fail, memcmp(sb.beg, bytes, sizeof(bytes)) == 0);
  // Trailing sentinel NUL after content.
  CHECK(Fail, sb.beg[sizeof(bytes)] == 0);
  strbuf_free(&sb);
  return 0;
Fail:
  strbuf_free(&sb);
  return 1;
}

int
main(void)
{
  int failed = 0;
  failed += test_zero_init_is_usable();
  failed += test_inline_append();
  failed += test_spill_to_heap();
  failed += test_appendf();
  failed += test_appendf_grows_past_inline();
  failed += test_reset_keeps_capacity();
  failed += test_set();
  failed += test_set_reuses_heap_allocation();
  failed += test_reset_on_zero_init();
  failed += test_reserve_zero();
  failed += test_appendf_at_inline_boundary();
  failed += test_reserve_overflow_guard();
  failed += test_binary_with_embedded_nul();
  if (failed)
    log_error("%d strbuf tests FAILED", failed);
  else
    log_info("all strbuf tests PASSED");
  return failed != 0;
}
