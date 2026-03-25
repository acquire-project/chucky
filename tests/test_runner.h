#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include "gpu/prelude.cuda.h"
#include "util/prelude.h"

// RUN_GPU_TESTS({"name", fn}, ...)
//   Replaces main() boilerplate: cuInit, cuDeviceGet, cuCtxCreate,
//   test dispatch loop (name + fn pointer pairs), cuCtxDestroy.
//   Returns 0 if all tests pass, 1 if any fail.

#define RUN_GPU_TESTS(...)                                                     \
  int main(int ac, char* av[])                                                 \
  {                                                                            \
    (void)ac;                                                                  \
    (void)av;                                                                  \
                                                                               \
    CUcontext ctx_ = 0;                                                        \
    CUdevice dev_;                                                             \
                                                                               \
    CU(RunGpuFail_, cuInit(0));                                                \
    CU(RunGpuFail_, cuDeviceGet(&dev_, 0));                                    \
    CU(RunGpuFail_, cuCtxCreate(&ctx_, 0, dev_));                              \
                                                                               \
    int rc_ = 0;                                                               \
    struct                                                                     \
    {                                                                          \
      const char* name;                                                        \
      int (*fn)(void);                                                         \
    } tests_[] = { __VA_ARGS__ };                                              \
    for (size_t i_ = 0; i_ < sizeof(tests_) / sizeof(tests_[0]); ++i_) {       \
      int r_ = tests_[i_].fn();                                                \
      if (r_) {                                                                \
        log_error("  FAIL: %s", tests_[i_].name);                              \
        rc_ = 1;                                                               \
      } else {                                                                 \
        log_info("  PASS: %s", tests_[i_].name);                               \
      }                                                                        \
    }                                                                          \
                                                                               \
    cuCtxDestroy(ctx_);                                                        \
    return rc_;                                                                \
                                                                               \
  RunGpuFail_:                                                                 \
    cuCtxDestroy(ctx_);                                                        \
    return 1;                                                                  \
  }

#endif // TEST_RUNNER_H
