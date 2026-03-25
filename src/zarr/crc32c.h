#pragma once

#include <stddef.h>
#include <stdint.h>

// Initialize CRC32C lookup table. Must be called before crc32c().
// Safe to call multiple times (no-op after first).
void
crc32c_init(void);

// Compute CRC32C (Castagnoli) checksum.
uint32_t
crc32c(const void* data, size_t len);
