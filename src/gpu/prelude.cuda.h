/// PRIVATE: never include in other headers.
#pragma once

#include "log/log.h"
#include <cuda.h>

#define CU(lbl, e)                                                             \
  do {                                                                         \
    CUresult res_ = (e);                                                       \
    if (res_ != CUDA_SUCCESS &&                                                \
        handle_curesult(LOG_ERROR, res_, __FILE__, __LINE__, #e)) {            \
      goto lbl;                                                                \
    }                                                                          \
  } while (0)

#define CUWARN(e)                                                              \
  do {                                                                         \
    handle_curesult(LOG_WARN, (e), __FILE__, __LINE__, #e);                    \
  } while (0)

#ifdef __cplusplus
extern "C"
{
#endif

  static inline int handle_curesult(int level,
                                    CUresult ecode,
                                    const char* file,
                                    int line,
                                    const char* expr)
  {
    if (ecode == CUDA_SUCCESS)
      return 0;
    const char *name, *desc;
    cuGetErrorName(ecode, &name);
    cuGetErrorString(ecode, &desc);
    if (name && desc) {
      log_log(level, file, line, "CUDA error: %s %s %s\n", name, desc, expr);
    } else {
      log_log(level,
              file,
              line,
              "%s. Failed to retrieve error info for CUresult: %d\n",
              expr,
              ecode);
    }
    return 1;
  }

  static inline void cu_event_destroy(CUevent e)
  {
    if (e)
      cuEventDestroy(e);
  }
  static inline void cu_stream_destroy(CUstream s)
  {
    if (s)
      cuStreamDestroy(s);
  }
  static inline void cu_mem_free(CUdeviceptr p)
  {
    if (p)
      cuMemFree(p);
  }
  static inline void cu_mem_freehost(void* p)
  {
    if (p)
      cuMemFreeHost(p);
  }

#ifdef __cplusplus
}
#endif
