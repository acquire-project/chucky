#include "chucky_log.h"
#include "util/prelude.h"
#include "util/strbuf.h"

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
  if (failed)
    log_error("%d strbuf tests FAILED", failed);
  else
    log_info("all strbuf tests PASSED");
  return failed != 0;
}
