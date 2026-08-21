#include "util/prelude.h"
#include "zarr/io_queue.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- test: ordering ---

struct order_ctx
{
  int* log;
  int index;
};

static void
order_fn(void* arg)
{
  struct order_ctx* c = (struct order_ctx*)arg;
  c->log[c->index] = c->index;
}

static int
test_ordering(void)
{
  struct io_queue* q = io_queue_create();
  CHECK(Fail, q);

  int log[100];
  memset(log, -1, sizeof(log));

  struct order_ctx ctxs[100];
  for (int i = 0; i < 100; ++i) {
    ctxs[i] = (struct order_ctx){ .log = log, .index = i };
    io_queue_post(q, (struct io_work){ .fn = order_fn, .ctx = &ctxs[i] });
  }

  struct io_event ev = io_queue_record(q);
  io_event_wait(q, ev);

  for (int i = 0; i < 100; ++i)
    CHECK(Fail2, log[i] == i);

  io_queue_destroy(q);
  return 0;

Fail2:
  io_queue_destroy(q);
Fail:
  return 1;
}

// --- test: event wait ---

static void
set_value(void* arg)
{
  atomic_int* val = (atomic_int*)arg;
  atomic_store(val, 1);
}

static int
test_event_wait(void)
{
  struct io_queue* q = io_queue_create();
  CHECK(Fail, q);

  atomic_int val = 0;
  io_queue_post(q, (struct io_work){ .fn = set_value, .ctx = (void*)&val });
  struct io_event ev = io_queue_record(q);
  io_event_wait(q, ev);

  CHECK(Fail2, atomic_load(&val) == 1);

  io_queue_destroy(q);
  return 0;

Fail2:
  io_queue_destroy(q);
Fail:
  return 1;
}

// --- test: ctx_free called ---

static void
noop_fn(void* arg)
{
  (void)arg;
}

static void
free_counter(void* arg)
{
  int* count = (int*)arg;
  (*count)++;
}

static int
test_ctx_free(void)
{
  struct io_queue* q = io_queue_create();
  CHECK(Fail, q);

  int free_count = 0;
  for (int i = 0; i < 10; ++i)
    io_queue_post(q,
                  (struct io_work){ .fn = noop_fn,
                                    .ctx = &free_count,
                                    .ctx_free = free_counter });

  struct io_event ev = io_queue_record(q);
  io_event_wait(q, ev);

  CHECK(Fail2, free_count == 10);

  io_queue_destroy(q);
  return 0;

Fail2:
  io_queue_destroy(q);
Fail:
  return 1;
}

// --- test: destroy drains ---

static void
increment(void* arg)
{
  atomic_int* val = (atomic_int*)arg;
  atomic_fetch_add(val, 1);
}

static int
test_destroy_drains(void)
{
  struct io_queue* q = io_queue_create();
  CHECK(Fail, q);

  atomic_int count = 0;
  for (int i = 0; i < 50; ++i)
    io_queue_post(q, (struct io_work){ .fn = increment, .ctx = (void*)&count });

  io_queue_destroy(q);
  CHECK(Fail, atomic_load(&count) == 50);
  return 0;

Fail:
  return 1;
}

// --- test: empty queue event ---

static int
test_empty_queue_event(void)
{
  struct io_queue* q = io_queue_create();
  CHECK(Fail, q);

  // Recording an event on an empty queue should return immediately
  struct io_event ev = io_queue_record(q);
  io_event_wait(q, ev);

  io_queue_destroy(q);
  return 0;

Fail:
  return 1;
}

// --- test: pending bytes ---

// #201: the reported figure has to account for every job the queue holds. The
// old split counter, raised after the post, could report fewer.

static void
wait_for_gate(void* arg)
{
  atomic_int* gate = (atomic_int*)arg;
  while (atomic_load(gate) == 0)
    ;
}

static int
test_pending_bytes(void)
{
  struct io_queue* q = io_queue_create();
  CHECK(Fail, q);

  CHECK(Fail2, io_queue_pending_bytes(q) == 0);

  // Hold the worker on the first job so the rest stay queued.
  atomic_int gate = 0;
  CHECK(Fail2,
        io_queue_post(
          q, (struct io_work){ .fn = wait_for_gate, .ctx = (void*)&gate }) ==
          0);

  uint64_t posted = 0;
  for (int i = 1; i <= 8; ++i) {
    uint64_t nbytes = (uint64_t)i * 1024;
    CHECK(Fail3,
          io_queue_post(
            q, (struct io_work){ .fn = noop_fn, .nbytes = nbytes }) == 0);
    posted += nbytes;
    CHECK(Fail3, io_queue_pending_bytes(q) == posted);
  }

  atomic_store(&gate, 1);
  io_event_wait(q, io_queue_record(q));
  CHECK(Fail2, io_queue_pending_bytes(q) == 0);

  // A job posted without a byte count leaves the figure alone.
  CHECK(Fail2, io_queue_post(q, (struct io_work){ .fn = noop_fn }) == 0);
  io_event_wait(q, io_queue_record(q));
  CHECK(Fail2, io_queue_pending_bytes(q) == 0);

  io_queue_destroy(q);
  return 0;

Fail3:
  atomic_store(&gate, 1);
Fail2:
  io_queue_destroy(q);
Fail:
  return 1;
}

// --- test: write stats ---

// Only queue depth available is measurable: one worker, so achieved is one.

static int
test_stats(void)
{
  struct io_queue* q = io_queue_create();
  CHECK(Fail, q);

  // Hold the worker so everything posted behind it stays waiting.
  atomic_int gate = 0;
  CHECK(Fail2,
        io_queue_post(
          q, (struct io_work){ .fn = wait_for_gate, .ctx = (void*)&gate }) ==
          0);

  // Three files, two writes each, round-robin: three only if files differ.
  for (int round = 0; round < 2; ++round)
    for (uint64_t file = 1; file <= 3; ++file)
      CHECK(Fail3,
            io_queue_post(q,
                          (struct io_work){ .fn = noop_fn,
                                            .nbytes = 4096,
                                            .file = file,
                                            .borrowed = (int)(file == 1) }) ==
              0);

  struct io_queue_stats st;
  io_queue_get_stats(q, &st);
  CHECK(Fail3, st.files_waiting_peak == 3);
  CHECK(Fail3, st.writes == 6);
  CHECK(Fail3, st.bytes_borrowed == 2 * 4096);
  CHECK(Fail3, st.bytes_copied == 4 * 4096);
  CHECK(Fail3, st.bytes_waiting_peak == 6 * 4096);
  CHECK(Fail3, st.size_buckets[12] == 6); // 4096 == 1 << 12

  // No file named: no change to the file count.
  CHECK(Fail3, io_queue_post(q, (struct io_work){ .fn = noop_fn }) == 0);
  io_queue_get_stats(q, &st);
  CHECK(Fail3, st.files_waiting_peak == 3);

  // Not counted for truncate and close: no payload.
  CHECK(Fail3,
        io_queue_post(q, (struct io_work){ .fn = noop_fn, .file = 4 }) == 0);
  CHECK(Fail3,
        io_queue_post(q, (struct io_work){ .fn = noop_fn, .file = 5 }) == 0);
  io_queue_get_stats(q, &st);
  CHECK(Fail3, st.files_waiting_peak == 3);

  atomic_store(&gate, 1);
  io_event_wait(q, io_queue_record(q));

  io_queue_get_stats(q, &st);
  // Peak is a high-water mark, safe from draining; the mean must be positive.
  CHECK(Fail2, st.files_waiting_peak == 3);
  CHECK(Fail2, st.files_waiting_mean > 0.0);
  // Every write was held behind the gate before it could start.
  CHECK(Fail2, st.wait_ms_max > 0.0);

  io_queue_destroy(q);
  return 0;

Fail3:
  atomic_store(&gate, 1);
Fail2:
  io_queue_destroy(q);
Fail:
  return 1;
}

// --- test: timing averages count finished writes ---

// Divisor is the finished count, not the posted count.

static int
test_timing_mean_counts_finished(void)
{
  struct io_queue* q = io_queue_create();
  CHECK(Fail, q);

  // One write, held behind a gate so its wait is measurable, then released.
  atomic_int gate = 0;
  CHECK(Fail2,
        io_queue_post(
          q, (struct io_work){ .fn = wait_for_gate, .ctx = (void*)&gate }) ==
          0);
  CHECK(Fail3,
        io_queue_post(q, (struct io_work){ .fn = noop_fn, .nbytes = 4096 }) ==
          0);
  atomic_store(&gate, 1);
  io_event_wait(q, io_queue_record(q));

  // Five more, still queued behind the second gate: jobs run in order.
  atomic_int held = 0;
  CHECK(Fail4,
        io_queue_post(
          q, (struct io_work){ .fn = wait_for_gate, .ctx = (void*)&held }) ==
          0);
  for (int i = 0; i < 5; ++i)
    CHECK(Fail4,
          io_queue_post(q, (struct io_work){ .fn = noop_fn, .nbytes = 4096 }) ==
            0);

  struct io_queue_stats st;
  io_queue_get_stats(q, &st);
  CHECK(Fail4, st.writes == 6);
  CHECK(Fail4, st.wait_ms_max > 0.0);
  // One finished write, so its wait is both the average and the maximum.
  CHECK(Fail4, st.wait_ms_mean == st.wait_ms_max);
  CHECK(Fail4, st.run_ms_mean == st.run_ms_max);

  atomic_store(&held, 1);
  io_queue_destroy(q);
  return 0;

Fail4:
  atomic_store(&held, 1);
Fail3:
  atomic_store(&gate, 1);
Fail2:
  io_queue_destroy(q);
Fail:
  return 1;
}

// --- main ---

int
main(void)
{
  int rc = 0;
  struct
  {
    const char* name;
    int (*fn)(void);
  } tests[] = {
    { "ordering", test_ordering },
    { "event_wait", test_event_wait },
    { "ctx_free", test_ctx_free },
    { "destroy_drains", test_destroy_drains },
    { "empty_queue_event", test_empty_queue_event },
    { "pending_bytes", test_pending_bytes },
    { "stats", test_stats },
    { "timing_mean_counts_finished", test_timing_mean_counts_finished },
  };
  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
    int r = tests[i].fn();
    if (r) {
      log_error("  FAIL: %s", tests[i].name);
      rc = 1;
    } else {
      log_info("  PASS: %s", tests[i].name);
    }
  }
  return rc;
}
