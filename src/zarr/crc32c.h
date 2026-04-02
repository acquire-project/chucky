#pragma once

#include <stddef.h>
#include <stdint.h>

// Compute CRC32C (Castagnoli) checksum.
// Self-initializing; safe to call from any thread.
uint32_t
crc32c(const void* data, size_t len);
