#include "platform/platform.h"

void
platform_touch_pages(void* ptr, size_t bytes)
{
  if (!ptr || !bytes)
    return;
  volatile unsigned char* data = ptr;
  const size_t page = platform_page_alignment();
  const size_t count = 1 + (bytes - 1) / page;
  for (size_t i = 0; i < count; ++i)
    data[i * page] = 0;
  data[bytes - 1] = 0;
}
