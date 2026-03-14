/// PRIVATE: never include in other headers.
#pragma once

#include "log/log.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define container_of(ptr, type, member)                                        \
  ((type*)((char*)(ptr) - offsetof(type, member)))

#define countof(e) (sizeof(e) / sizeof(e[0]))

#define CHECK(lbl, e)                                                          \
  do {                                                                         \
    if (!(e)) {                                                                \
      log_error("%s:%d check failed: %s", __FILE__, __LINE__, #e);             \
      goto lbl;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_SILENT(lbl, e)                                                   \
  do {                                                                         \
    if (!(e))                                                                  \
      goto lbl;                                                                \
  } while (0)

  static inline uint64_t ceildiv(uint64_t a, uint64_t b)
  {
    return (a + b - 1) / b;
  }

  static inline size_t align_up(size_t x, size_t alignment)
  {
    return (x + alignment - 1) / alignment * alignment;
  }

  static inline int ceil_log2(uint64_t v)
  {
    int p = 0;
    while ((1ull << p) < v)
      ++p;
    return p;
  }

#ifdef __cplusplus
}
#endif
