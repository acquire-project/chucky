/// PRIVATE: never include in other headers.
#pragma once

#include "log/log.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

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

// Wraps a kernel launch. A launch the driver turns away queues nothing and
// leaves its reason in the runtime's per-thread error, which the driver-API
// calls around it never see. Yields 0 when the launch was accepted.
//
// Taking the pending error first is what scopes the check to this launch:
// nvcomp runs kernels on this same host thread, and a launch that succeeds
// leaves an error already stored there for this check to find. A fault that
// poisoned the context still reports, since the read after the launch returns
// it again.
#define CUDA_LAUNCH(...)                                                       \
  (warn_stale_cudaerror(__FILE__, __LINE__),                                   \
   (__VA_ARGS__),                                                              \
   handle_cudaerror(cudaGetLastError(), __FILE__, __LINE__, #__VA_ARGS__))

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

  static inline int handle_cudaerror(cudaError_t ecode,
                                     const char* file,
                                     int line,
                                     const char* expr)
  {
    if (ecode == cudaSuccess)
      return 0;
    log_log(LOG_ERROR,
            file,
            line,
            "CUDA error: %s %s\n",
            cudaGetErrorString(ecode),
            expr);
    return 1;
  }

  // Takes the error another runtime user left unread. Reporting it here is
  // the only account of it anyone gets: whoever produced it returned success
  // to its own caller, and clearing is what keeps it from being read back as
  // the next launch's failure.
  static inline void warn_stale_cudaerror(const char* file, int line)
  {
    const cudaError_t stale = cudaGetLastError();
    if (stale != cudaSuccess)
      log_log(LOG_WARN,
              file,
              line,
              "CUDA error left by an earlier call on this thread: %s\n",
              cudaGetErrorString(stale));
  }

  // Kernel launches go through the runtime API, which uses the calling
  // thread's context and refuses a stream that belongs to another one, so an
  // entry point a caller can reach from any thread makes ctx current first.
  // Pushing rather than setting leaves a caller that keeps its own context
  // holding it again on return. Yields what to hand cu_ctx_pop.
  static inline int cu_ctx_push(CUcontext ctx)
  {
    CUcontext current = NULL;
    if (!ctx || (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == ctx))
      return 0;
    const CUresult res = cuCtxPushCurrent(ctx);
    // Say so rather than let the caller read the 0 as "already current" and
    // run the whole append against whatever context this thread does hold.
    if (res != CUDA_SUCCESS) {
      handle_curesult(LOG_ERROR, res, __FILE__, __LINE__, "cuCtxPushCurrent");
      return 0;
    }
    return 1;
  }

  static inline void cu_ctx_pop(int pushed)
  {
    CUcontext prev = NULL;
    if (pushed)
      CUWARN(cuCtxPopCurrent(&prev));
  }

  // CUDA 13 added a CUctxCreateParams* argument in position 2; pass NULL to
  // preserve the CUDA 12 behaviour. Wrap so call sites stay portable.
  static inline CUresult cu_ctx_create(CUcontext* pctx,
                                       unsigned int flags,
                                       CUdevice dev)
  {
#if CUDA_VERSION >= 13000
    return cuCtxCreate(pctx, NULL, flags, dev);
#else
  return cuCtxCreate(pctx, flags, dev);
#endif
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
  static inline void cu_stream_sync(CUstream s)
  {
    if (s)
      cuStreamSynchronize(s);
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
