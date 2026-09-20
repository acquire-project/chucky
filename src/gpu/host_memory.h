#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Allocate zeroed host memory, then register it for CUDA transfers. The size
  // is rounded up to alignment, which must be a power of two >= the page size.
  // First touch uses the calling thread's current memory policy.
  void* gpu_host_alloc(size_t alignment, size_t bytes);

  // Unregister and release memory from gpu_host_alloc. NULL is harmless.
  void gpu_host_free(void* data);

#ifdef __cplusplus
}
#endif
