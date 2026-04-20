#include "util/format_bytes.h"

#include <stdio.h>

void
format_bytes(char* buf, size_t bufsz, uint64_t bytes)
{
  if (!buf || bufsz == 0)
    return;

  const double KIB = 1024.0;
  const double MIB = 1024.0 * 1024.0;
  const double GIB = 1024.0 * 1024.0 * 1024.0;

  if (bytes >= (uint64_t)GIB)
    snprintf(buf, bufsz, "%.2f GiB", (double)bytes / GIB);
  else if (bytes >= (uint64_t)MIB)
    snprintf(buf, bufsz, "%.2f MiB", (double)bytes / MIB);
  else if (bytes >= (uint64_t)KIB)
    snprintf(buf, bufsz, "%.2f KiB", (double)bytes / KIB);
  else
    snprintf(buf, bufsz, "%llu B", (unsigned long long)bytes);
}
