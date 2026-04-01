#include "zarr/crc32c.h"

#include <stdatomic.h>

static uint32_t crc32c_table[256];
static _Atomic int crc32c_table_ready;

static void
crc32c_init(void)
{
  for (int i = 0; i < 256; ++i) {
    uint32_t crc = (uint32_t)i;
    for (int j = 0; j < 8; ++j)
      crc = (crc >> 1) ^ (0x82F63B78 & (0u - (crc & 1)));
    crc32c_table[i] = crc;
  }
  atomic_store_explicit(&crc32c_table_ready, 1, memory_order_release);
}

uint32_t
crc32c(const void* data, size_t len)
{
  if (!atomic_load_explicit(&crc32c_table_ready, memory_order_acquire))
    crc32c_init();
  uint32_t crc = 0xFFFFFFFF;
  const uint8_t* p = (const uint8_t*)data;
  for (size_t i = 0; i < len; ++i)
    crc = crc32c_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFF;
}
