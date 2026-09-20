#include "gpu/host_memory.h"

#include "gpu/prelude.cuda.h"
#include "platform/platform.h"

#include <stdint.h>
#include <string.h>

void*
gpu_host_alloc(size_t alignment, size_t bytes)
{
  if (alignment < platform_page_alignment() || (alignment & (alignment - 1)) ||
      bytes == 0 || bytes > SIZE_MAX - (alignment - 1))
    return NULL;
  bytes = (bytes + alignment - 1) & ~(alignment - 1);
  void* data = platform_aligned_alloc(alignment, bytes);
  if (!data)
    return NULL;
  // Touch before registration, while the caller's placement scope is active.
  memset(data, 0, bytes);
  const CUresult result = cuMemHostRegister(data, bytes, 0);
  if (handle_curesult(
        LOG_ERROR, result, __FILE__, __LINE__, "cuMemHostRegister")) {
    platform_aligned_free(data);
    return NULL;
  }
  return data;
}

void
gpu_host_free(void* data)
{
  if (!data)
    return;
  handle_curesult(LOG_ERROR,
                  cuMemHostUnregister(data),
                  __FILE__,
                  __LINE__,
                  "cuMemHostUnregister");
  platform_aligned_free(data);
}
