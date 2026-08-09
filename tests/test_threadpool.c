// Unit tests for the chucky thread pool. Covers create/destroy, the three
// dispatch primitives (for_n, for_n_dynamic, for_threads), repeated dispatch
// (exercises wake-up), and idle-then-dispatch (exercises condvar sleep).

#include "platform/platform.h"
#include "threadpool/threadpool.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(label, expr)                                                     \
  do {                                                                         \
    if (!(expr)) {                                                             \
      fprintf(                                                                 \
        stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr);     \
      goto label;                                                              \
    }                                                                          \
  } while (0)

// ---- for_n: static slice coverage ------------------------------------------

struct for_n_ctx
{
  _Atomic uint64_t sum;
  _Atomic uint64_t* hits; // hits[i] should be exactly 1 for each i
};

static void
for_n_body(size_t beg, size_t end, int tid, void* vctx)
{
  (void)tid;
  struct for_n_ctx* c = (struct for_n_ctx*)vctx;
  uint64_t local = 0;
  for (size_t i = beg; i < end; ++i) {
    local += i;
    atomic_fetch_add_explicit(&c->hits[i], 1, memory_order_relaxed);
  }
  atomic_fetch_add_explicit(&c->sum, local, memory_order_relaxed);
}

static int
test_for_n(struct threadpool* p, size_t n)
{
  int rc = -1;
  _Atomic uint64_t* hits = (_Atomic uint64_t*)calloc(n ? n : 1, sizeof(*hits));
  CHECK(Done, hits || n == 0);
  struct for_n_ctx c = { 0, hits };

  threadpool_for_n(p, n, for_n_body, &c);

  uint64_t expected = (uint64_t)n * ((uint64_t)n - 1) / 2;
  if (n == 0)
    expected = 0;
  uint64_t got = atomic_load_explicit(&c.sum, memory_order_acquire);
  CHECK(Done, got == expected);

  for (size_t i = 0; i < n; ++i) {
    uint64_t h = atomic_load_explicit(&hits[i], memory_order_acquire);
    CHECK(Done, h == 1);
  }
  rc = 0;
Done:
  free((void*)hits);
  return rc;
}

// ---- for_n_dynamic: each index runs exactly once ---------------------------

struct for_n_dyn_ctx
{
  _Atomic uint64_t* hits;
};

static void
for_n_dyn_body(size_t i, int tid, void* vctx)
{
  (void)tid;
  struct for_n_dyn_ctx* c = (struct for_n_dyn_ctx*)vctx;
  atomic_fetch_add_explicit(&c->hits[i], 1, memory_order_relaxed);
}

static int
test_for_n_dynamic(struct threadpool* p, size_t n)
{
  int rc = -1;
  _Atomic uint64_t* hits = (_Atomic uint64_t*)calloc(n ? n : 1, sizeof(*hits));
  CHECK(Done, hits || n == 0);
  struct for_n_dyn_ctx c = { hits };

  threadpool_for_n_dynamic(p, n, for_n_dyn_body, &c);

  for (size_t i = 0; i < n; ++i) {
    uint64_t h = atomic_load_explicit(&hits[i], memory_order_acquire);
    CHECK(Done, h == 1);
  }
  rc = 0;
Done:
  free((void*)hits);
  return rc;
}

// ---- for_threads: each tid called exactly once -----------------------------

struct for_threads_ctx
{
  _Atomic uint64_t* tid_hits;
  _Atomic int observed_nthreads;
};

static void
for_threads_body(int tid, int nthreads, void* vctx)
{
  struct for_threads_ctx* c = (struct for_threads_ctx*)vctx;
  atomic_fetch_add_explicit(&c->tid_hits[tid], 1, memory_order_relaxed);
  atomic_store_explicit(&c->observed_nthreads, nthreads, memory_order_relaxed);
}

static int
test_for_threads(struct threadpool* p)
{
  int rc = -1;
  int nt = threadpool_size(p);
  _Atomic uint64_t* hits = (_Atomic uint64_t*)calloc((size_t)nt, sizeof(*hits));
  CHECK(Done, hits);
  struct for_threads_ctx c = { hits, 0 };

  threadpool_for_threads(p, for_threads_body, &c);

  CHECK(Done,
        atomic_load_explicit(&c.observed_nthreads, memory_order_acquire) == nt);
  for (int i = 0; i < nt; ++i) {
    uint64_t h = atomic_load_explicit(&hits[i], memory_order_acquire);
    CHECK(Done, h == 1);
  }
  rc = 0;
Done:
  free((void*)hits);
  return rc;
}

// ---- Stress: many dispatches in a row (wake-up path) -----------------------

struct stress_ctx
{
  _Atomic uint64_t counter;
};

static void
stress_body(size_t beg, size_t end, int tid, void* vctx)
{
  (void)tid;
  struct stress_ctx* c = (struct stress_ctx*)vctx;
  atomic_fetch_add_explicit(
    &c->counter, (uint64_t)(end - beg), memory_order_relaxed);
}

static int
test_stress(struct threadpool* p, int iters, size_t per)
{
  int rc = -1;
  struct stress_ctx c = { 0 };
  for (int i = 0; i < iters; ++i)
    threadpool_for_n(p, per, stress_body, &c);
  uint64_t got = atomic_load_explicit(&c.counter, memory_order_acquire);
  CHECK(Done, got == (uint64_t)iters * per);
  rc = 0;
Done:
  return rc;
}

// ---- Idle-then-dispatch (condvar sleep path) -------------------------------

static int
test_idle_then_dispatch(struct threadpool* p)
{
  // Sleep long enough that workers definitely fall through their spin budget
  // and call cond_wait.
  platform_sleep_ns(200 * 1000 * 1000); // 200 ms
  return test_for_n(p, 4096);
}

// ---- Driver ----------------------------------------------------------------

int
main(void)
{
  int failed = 0;

  // Pool sizes to exercise: 0 workers (caller-only serial), 1 worker, 8.
  int sizes[] = { 0, 1, 8 };
  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
    int nworkers = sizes[s];
    struct threadpool* p = threadpool_new(nworkers);
    if (!p) {
      fprintf(stderr, "threadpool_new(%d) failed\n", nworkers);
      failed = 1;
      continue;
    }

    size_t coverage[] = { 0, 1, 7, 1024, 1u << 20 };
    for (size_t k = 0; k < sizeof(coverage) / sizeof(coverage[0]); ++k) {
      if (test_for_n(p, coverage[k]) != 0) {
        fprintf(
          stderr, "for_n nworkers=%d n=%zu FAILED\n", nworkers, coverage[k]);
        failed = 1;
      }
      if (test_for_n_dynamic(p, coverage[k]) != 0) {
        fprintf(stderr,
                "for_n_dynamic nworkers=%d n=%zu FAILED\n",
                nworkers,
                coverage[k]);
        failed = 1;
      }
    }

    if (test_for_threads(p) != 0) {
      fprintf(stderr, "for_threads nworkers=%d FAILED\n", nworkers);
      failed = 1;
    }

    if (test_stress(p, 100000, 16) != 0) {
      fprintf(stderr, "stress nworkers=%d FAILED\n", nworkers);
      failed = 1;
    }

    if (nworkers > 0 && test_idle_then_dispatch(p) != 0) {
      fprintf(stderr, "idle_then_dispatch nworkers=%d FAILED\n", nworkers);
      failed = 1;
    }

    threadpool_free(p);
  }

  if (failed) {
    fprintf(stderr, "FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}
