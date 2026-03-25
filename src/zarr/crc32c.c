#include "zarr/crc32c.h"

static uint32_t crc32c_table[256];
static int crc32c_table_ready;

void
crc32c_init(void)
{
  if (crc32c_table_ready)
    return;
  for (int i = 0; i < 256; ++i) {
    uint32_t crc = (uint32_t)i;
    for (int j = 0; j < 8; ++j)
      crc = (crc >> 1) ^ (0x82F63B78 & (0u - (crc & 1)));
    crc32c_table[i] = crc;
  }
  crc32c_table_ready = 1;
}

uint32_t
crc32c(const void* data, size_t len)
{
  uint32_t crc = 0xFFFFFFFF;
  const uint8_t* p = (const uint8_t*)data;
  for (size_t i = 0; i < len; ++i)
    crc = crc32c_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFF;
}
