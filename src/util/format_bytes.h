#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Format a byte count into a human-readable string with a power-of-two
  // suffix (B, KiB, MiB, GiB). Writes up to bufsz-1 chars plus a NUL into
  // buf. The unit is chosen by magnitude: >= 1 GiB -> GiB, >= 1 MiB -> MiB,
  // >= 1 KiB -> KiB, else bytes. Thread-safe: caller owns the buffer.
  //
  // A 32-byte buffer is always enough.
  void format_bytes(char* buf, size_t bufsz, uint64_t bytes);

#ifdef __cplusplus
}
#endif
