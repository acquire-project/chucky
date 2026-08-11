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
  (take_stale_cudaerror(__FILE__, __LINE__),                                   \
   (__VA_ARGS__),                                                              \
   handle_cudaerror(cudaGetLastError(), __FILE__, __LINE__, #__VA_ARGS__))

// Wraps a runtime-API call that returns its own status. Yields 0 on success.
#define CUDA_CALL(e) handle_cudaerror((e), __FILE__, __LINE__, #e)

// The same two, in the goto form CU uses, for the sites that clean up.
#define CUDA_LAUNCH_OR(lbl, ...)                                               \
  do {                                                                         \
    if (CUDA_LAUNCH(__VA_ARGS__))                                              \
      goto lbl;                                                                \
  } while (0)

#define CUDA_CALL_OR(lbl, e)                                                   \
  do {                                                                         \
    if (CUDA_CALL(e))                                                          \
      goto lbl;                                                                \
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

  // Takes the error another runtime user left unread, so the check after the
  // launch answers for the launch alone. Debug rather than warn: a sticky
  // fault is returned again by every later call, so warning here would print
  // once per launch for the rest of the run, and the launch's own check
  // reports that fault anyway.
  static inline void take_stale_cudaerror(const char* file, int line)
  {
    const cudaError_t stale = cudaGetLastError();
    if (stale != cudaSuccess)
      log_log(LOG_DEBUG,
              file,
              line,
              "CUDA error left by an earlier call on this thread: %s\n",
              cudaGetErrorString(stale));
  }

  // Kernel launches go through the runtime API, which uses the calling
  // thread's context and refuses a stream that belongs to another one, so an
  // entry point a caller can reach from any thread makes ctx current first.
  // Pushing rather than setting leaves a caller that keeps its own context
  // holding it again on return.
  //
  // Yields 1 when it pushed, 0 when the thread already holds ctx, and -1 when
  // the push failed. Work done after a -1 would run against whatever context
  // the thread does hold, so callers that can report an error should.
  static inline int cu_ctx_push(CUcontext ctx)
  {
    CUcontext current = NULL;
    if (!ctx || (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == ctx))
      return 0;
    const CUresult res = cuCtxPushCurrent(ctx);
    if (res != CUDA_SUCCESS) {
      handle_curesult(LOG_ERROR, res, __FILE__, __LINE__, "cuCtxPushCurrent");
      return -1;
    }
    return 1;
  }

  static inline void cu_ctx_pop(int pushed)
  {
    if (pushed == 1) {
      CUcontext prev = NULL;
      CUWARN(cuCtxPopCurrent(&prev));
    }
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
