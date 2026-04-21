// Dynamic string buffer with small-string inline storage.
//
// Use for path/key construction and other variable-length text where a
// fixed-size stack buffer would risk silent truncation. Short content
// (<= STRBUF_INLINE_CAP bytes) stays in the inline buffer — zero heap
// allocations. Growth spills to heap, then reuses the allocation across
// subsequent reset/append cycles.
//
// Zero-initialization is valid: `struct strbuf sb = {0}` is safe to pass
// to any function; the inline buffer lazy-activates on first append.
//
// All append functions return 0 on success, non-zero on allocation
// failure. Content and NUL terminator are always kept consistent: after a
// successful call, `*sb.end == 0` and `strbuf_cstr(&sb)` is a valid C
// string.
#pragma once

#include <stdarg.h>
#include <stddef.h>

#define STRBUF_INLINE_CAP 128

struct strbuf
{
  char inline_buf[STRBUF_INLINE_CAP];
  char* beg;     // NULL until first use; then points at inline_buf or heap
  char* end;     // *end == 0; len == end - beg
  char* cap_end; // one past last writable byte (NUL slot)
};

// Free heap allocation (if any). Safe on zero-init and already-freed bufs.
// Leaves sb in a re-usable zero-init state.
void
strbuf_free(struct strbuf* sb);

// Set length to 0, keep allocation. Safe on zero-init (no-op).
void
strbuf_reset(struct strbuf* sb);

// Ensure there is room for `need` more content bytes plus the NUL. Lazily
// activates inline storage on first call. Returns 0 on success.
int
strbuf_reserve(struct strbuf* sb, size_t need);

// Append raw bytes. Returns 0 on success.
int
strbuf_append(struct strbuf* sb, const char* data, size_t n);

// Append a NUL-terminated C string. Returns 0 on success.
int
strbuf_append_cstr(struct strbuf* sb, const char* s);

// printf-style append. Returns 0 on success.
int
strbuf_appendf(struct strbuf* sb, const char* fmt, ...)
  __attribute__((format(printf, 2, 3)));

// va_list variant of strbuf_appendf.
int
strbuf_vappendf(struct strbuf* sb, const char* fmt, va_list ap);

// Reset then append the given C string (convenience for "replace contents").
// Returns 0 on success.
int
strbuf_set(struct strbuf* sb, const char* s);

// Current content length (bytes, excluding NUL).
static inline size_t
strbuf_len(const struct strbuf* sb)
{
  return sb->beg ? (size_t)(sb->end - sb->beg) : 0;
}

// NUL-terminated pointer to content. Never NULL: returns "" for empty/
// zero-init bufs.
const char*
strbuf_cstr(const struct strbuf* sb);
