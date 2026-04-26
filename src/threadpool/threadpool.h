#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // A small fixed-size thread pool for chucky's CPU streaming pipeline.
  // Replaces OpenMP. Workers spin briefly when idle, then sleep on a condvar
  // to keep idle CPU at zero between dispatches.
  struct threadpool;

  // Create a pool with `nthreads` workers (excluding the calling thread).
  // The caller thread participates in dispatch as tid 0; workers occupy
  // tids 1..nthreads. Total parallelism = nthreads + 1.
  // nthreads must be >= 0. nthreads == 0 is valid (everything runs serially
  // on the caller).
  // Returns NULL on failure.
  struct threadpool* threadpool_new(int nthreads);

  void threadpool_free(struct threadpool* p);

  // Total parallelism = worker count + 1 (the caller). NULL pool counts as 1.
  int threadpool_size(const struct threadpool* p);

  // All `threadpool_for_*` functions accept NULL pool, in which case the
  // body runs sequentially on the caller thread. This makes it cheap to
  // wire pools through call paths that don't always need parallelism.

  // --- Static parallel for: divide [0, n) into `threadpool_size(p)` slices,
  // call fn(beg, end, tid, ctx) on each. fn may safely capture ctx. ---
  typedef void (*threadpool_range_fn)(size_t beg,
                                      size_t end,
                                      int tid,
                                      void* ctx);

  void threadpool_for_n(struct threadpool* p,
                        size_t n,
                        threadpool_range_fn fn,
                        void* ctx);

  // --- Dynamic parallel for: each worker repeatedly fetch_add's a shared
  // counter to claim a single index in [0, n) until exhausted. Use when per-
  // index work varies (compress, etc.). fn called with one index at a time. ---
  typedef void (*threadpool_index_fn)(size_t i, int tid, void* ctx);

  void threadpool_for_n_dynamic(struct threadpool* p,
                                size_t n,
                                threadpool_index_fn fn,
                                void* ctx);

  // --- Broadcast: call fn(tid, nthreads, ctx) once per participant.
  // Replaces `#pragma omp parallel { tid = omp_get_thread_num(); ... }`. ---
  typedef void (*threadpool_broadcast_fn)(int tid, int nthreads, void* ctx);

  void threadpool_for_threads(struct threadpool* p,
                              threadpool_broadcast_fn fn,
                              void* ctx);

#ifdef __cplusplus
}
#endif
