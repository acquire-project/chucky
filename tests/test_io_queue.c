#include "platform/platform.h"
#include "test_io_backend_fake.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr/io_queue.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

// --- test: ordering ---

static int
test_ordering(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){ 0 });
  CHECK(Fail, q);

  for (uint64_t i = 0; i < 100; ++i)
    CHECK(Fail2,
          io_queue_post(q,
                        (struct io_request){ .op = IO_OP_WRITE,
                                             .file = { .generation = 1 },
                                             .nbytes = i + 1 }) == 0);

  io_event_wait(q, io_queue_record(q));

  CHECK(Fail2, io_backend_fake_record_count(&fake) == 100);
  for (uint64_t i = 0; i < 100; ++i) {
    CHECK(Fail2, fake.records[i].nbytes == i + 1);
    CHECK(Fail2, fake.records[i].seq == i + 1);
  }

  io_queue_destroy(q);
  return 0;

Fail2:
  io_queue_destroy(q);
Fail:
  return 1;
}

// --- test: event wait ---

static int
test_event_wait(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){ 0 });
  CHECK(Fail, q);

  CHECK(Fail2, io_queue_post(q, (struct io_request){ .op = IO_OP_NOOP }) == 0);
  struct io_event ev = io_queue_record(q);
  io_event_wait(q, ev);

  CHECK(Fail2, io_backend_fake_record_count(&fake) == 1);

  io_queue_destroy(q);
  return 0;

Fail2:
  io_queue_destroy(q);
Fail:
  return 1;
}

// --- test: owned payload released ---

struct owned_block
{
  _Atomic int* released;
};

static void
release_block(void* p)
{
  struct owned_block* b = (struct owned_block*)p;
  atomic_fetch_add(b->released, 1);
  free(b);
}

static int
test_owned_payload_released(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){ 0 });
  CHECK(Fail, q);

  _Atomic int released;
  atomic_store(&released, 0);

  for (int i = 0; i < 10; ++i) {
    struct owned_block* b = (struct owned_block*)malloc(sizeof(*b));
    CHECK(Fail2, b);
    b->released = &released;
    CHECK(Fail2,
          io_queue_post(q,
                        (struct io_request){ .op = IO_OP_WRITE,
                                             .file = { .generation = 1 },
                                             .payload = b,
                                             .nbytes = sizeof(*b),
                                             .owned = b,
                                             .owned_free = release_block }) ==
            0);
  }

  // The release happens after the request retires, outside the lock, so only
  // the join settles the count.
  io_queue_destroy(q);
  CHECK(Fail, atomic_load(&released) == 10);
  return 0;

Fail2:
  io_queue_destroy(q);
Fail:
  return 1;
}

// --- test: destroy drains ---

static int
test_destroy_drains(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){ 0 });
  CHECK(Fail, q);

  for (int i = 0; i < 50; ++i)
    CHECK(Fail2,
          io_queue_post(q, (struct io_request){ .op = IO_OP_NOOP }) == 0);

  io_queue_destroy(q);
  CHECK(Fail, io_backend_fake_record_count(&fake) == 50);
  return 0;

Fail2:
  io_queue_destroy(q);
Fail:
  return 1;
}

// --- test: empty queue event ---

static int
test_empty_queue_event(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){ 0 });
  CHECK(Fail, q);

  // Recording an event on an empty queue should return immediately
  struct io_event ev = io_queue_record(q);
  io_event_wait(q, ev);

  CHECK(Fail2, io_backend_fake_record_count(&fake) == 0);

  io_queue_destroy(q);
  return 0;

Fail2:
  io_queue_destroy(q);
Fail:
  return 1;
}

// --- test: pending bytes ---

// #201: the reported figure has to account for every job the queue holds. The
// old split counter, raised after the post, could report fewer.

static int
test_pending_bytes(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){ 0 });
  CHECK(Fail, q);

  CHECK(Fail2, io_queue_pending_bytes(q) == 0);

  // Hold the worker on the first request so the rest stay queued.
  _Atomic int gate;
  atomic_store(&gate, 0);
  io_backend_fake_hold(&fake, &gate);
  CHECK(Fail2, io_queue_post(q, (struct io_request){ .op = IO_OP_NOOP }) == 0);

  uint64_t posted = 0;
  for (uint64_t i = 1; i <= 8; ++i) {
    const uint64_t nbytes = i * 1024;
    CHECK(Fail3,
          io_queue_post(q,
                        (struct io_request){ .op = IO_OP_WRITE,
                                             .file = { .generation = 1 },
                                             .nbytes = nbytes }) == 0);
    posted += nbytes;
    CHECK(Fail3, io_queue_pending_bytes(q) == posted);
  }

  atomic_store(&gate, 1);
  io_event_wait(q, io_queue_record(q));
  CHECK(Fail2, io_queue_pending_bytes(q) == 0);

  // A request posted without a byte count leaves the figure alone.
  CHECK(Fail2, io_queue_post(q, (struct io_request){ .op = IO_OP_NOOP }) == 0);
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
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){ 0 });
  CHECK(Fail, q);

  // Hold the worker so everything posted behind it stays waiting.
  _Atomic int gate;
  atomic_store(&gate, 0);
  io_backend_fake_hold(&fake, &gate);
  CHECK(Fail2, io_queue_post(q, (struct io_request){ .op = IO_OP_NOOP }) == 0);

  // Three files, two writes each, round-robin: three only if files differ.
  for (int round = 0; round < 2; ++round)
    for (uint64_t file = 1; file <= 3; ++file)
      CHECK(Fail3,
            io_queue_post(
              q,
              (struct io_request){ .op = IO_OP_WRITE,
                                   .nbytes = 4096,
                                   .file = { .generation = file },
                                   .borrowed = (uint8_t)(file == 1) }) == 0);

  struct io_queue_stats st;
  io_queue_get_stats(q, &st);
  CHECK(Fail3, st.files_waiting_peak == 3);
  CHECK(Fail3, st.writes == 6);
  CHECK(Fail3, st.bytes_borrowed == 2 * 4096);
  CHECK(Fail3, st.bytes_copied == 4 * 4096);
  CHECK(Fail3, st.bytes_waiting_peak == 6 * 4096);
  CHECK(Fail3, st.size_buckets[12] == 6); // 4096 == 1 << 12

  // No file named: no change to the file count.
  CHECK(Fail3, io_queue_post(q, (struct io_request){ .op = IO_OP_NOOP }) == 0);
  io_queue_get_stats(q, &st);
  CHECK(Fail3, st.files_waiting_peak == 3);

  // Not counted for truncate and close: no payload.
  CHECK(Fail3,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_TRUNCATE,
                                           .file = { .generation = 4 },
                                           .logical_size = 4096 }) == 0);
  CHECK(Fail3,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_CLOSE,
                                           .file = { .generation = 5 } }) == 0);
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
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){ 0 });
  CHECK(Fail, q);

  // One write, held behind a gate so its wait is measurable, then released.
  _Atomic int gate;
  atomic_store(&gate, 0);
  io_backend_fake_hold(&fake, &gate);
  CHECK(Fail2, io_queue_post(q, (struct io_request){ .op = IO_OP_NOOP }) == 0);
  CHECK(Fail3,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .file = { .generation = 1 },
                                           .nbytes = 4096 }) == 0);
  atomic_store(&gate, 1);
  io_event_wait(q, io_queue_record(q));

  // Five more, still queued behind the second gate: requests run in order.
  _Atomic int held;
  atomic_store(&held, 0);
  io_backend_fake_hold(&fake, &held);
  CHECK(Fail4, io_queue_post(q, (struct io_request){ .op = IO_OP_NOOP }) == 0);
  for (int i = 0; i < 5; ++i)
    CHECK(Fail4,
          io_queue_post(q,
                        (struct io_request){ .op = IO_OP_WRITE,
                                             .file = { .generation = 1 },
                                             .nbytes = 4096 }) == 0);

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

// --- shared helpers for the deferring cases ---

#define HANDOVER_TIMEOUT_MS 5000
#define HOLD_OBSERVE_MS 100
#define WRITE_BYTES 4096

// Poll until the fake has been handed n requests. Zero once it has, -1 if the
// wait ran out.
static int
wait_for_records(const struct io_backend_fake* f, uint64_t n, int timeout_ms)
{
  int waited_ms = 0;
  while (io_backend_fake_record_count(f) < n) {
    if (waited_ms >= timeout_ms)
      return -1;
    platform_sleep_ns(1000000LL);
    waited_ms += 1;
  }
  return 0;
}

static void
answer(struct io_queue* q, uint64_t* answered, struct io_completion c)
{
  *answered |= (uint64_t)1 << c.seq;
  io_queue_complete(q, c);
}

// Every exit from a deferring case comes through here: destroy blocks until
// each deferred request has been answered.
static void
answer_the_rest(struct io_queue* q,
                struct io_backend_fake* f,
                uint64_t* answered)
{
  // A barrier freed by one of these answers has to finish on its own.
  io_backend_fake_defer(f, 0);
  const uint64_t n = io_backend_fake_deferred_count(f);
  for (uint64_t i = 0; i < n; ++i) {
    const uint64_t seq = f->deferred[i];
    if (*answered & ((uint64_t)1 << seq))
      continue;
    answer(q, answered, (struct io_completion){ .seq = seq });
  }
}

struct fence_wait
{
  struct io_queue* q;
  struct io_event ev;
  _Atomic int entered;
  _Atomic int done;
};

static void
fence_wait_fn(void* arg)
{
  struct fence_wait* w = (struct fence_wait*)arg;
  atomic_store(&w->entered, 1);
  io_event_wait(w->q, w->ev);
  atomic_store(&w->done, 1);
}

// Waiting on another thread turns a watermark that never moves into a failure
// rather than a wedged test.
static int
fence_wait_start(struct fence_wait* w,
                 test_thread** thr,
                 struct io_queue* q,
                 struct io_event ev)
{
  w->q = q;
  w->ev = ev;
  atomic_store(&w->entered, 0);
  atomic_store(&w->done, 0);
  return test_thread_start(thr, fence_wait_fn, w);
}

// --- test: out-of-order retirement keeps the watermark ---

#define OUT_OF_ORDER_COUNT 4

static int
test_out_of_order_retirement(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  io_backend_fake_defer(&fake, 1);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){ 0 });
  CHECK(Fail, q);

  int rc = 1;
  uint64_t answered = 0;
  test_thread* thr = NULL;
  struct fence_wait w;

  CHECK(Cleanup,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .file = { .generation = 1 },
                                           .nbytes = WRITE_BYTES }) == 0);
  const struct io_event ev_first = io_queue_record(q);

  for (uint64_t i = 1; i < OUT_OF_ORDER_COUNT; ++i)
    CHECK(Cleanup,
          io_queue_post(q,
                        (struct io_request){ .op = IO_OP_WRITE,
                                             .file = { .generation = i + 1 },
                                             .nbytes = WRITE_BYTES }) == 0);
  const struct io_event ev_all = io_queue_record(q);

  // Each request names a different file, so none is held behind another.
  CHECK(Cleanup,
        wait_for_records(&fake, OUT_OF_ORDER_COUNT, HANDOVER_TIMEOUT_MS) == 0);
  CHECK(Cleanup, io_backend_fake_deferred_count(&fake) == OUT_OF_ORDER_COUNT);

  CHECK(Cleanup, fence_wait_start(&w, &thr, q, ev_first) == 0);
  CHECK(Cleanup, test_wait_flag(&w.entered, HANDOVER_TIMEOUT_MS) == 0);

  for (uint64_t seq = OUT_OF_ORDER_COUNT; seq > 1; --seq)
    answer(q,
           &answered,
           (struct io_completion){ .seq = seq, .nbytes = WRITE_BYTES });

  // Three answered and the oldest not, so the watermark has not moved.
  CHECK(Cleanup, test_wait_flag(&w.done, HOLD_OBSERVE_MS) == -1);
  CHECK(Cleanup, io_queue_pending_bytes(q) == WRITE_BYTES);

  answer(
    q, &answered, (struct io_completion){ .seq = 1, .nbytes = WRITE_BYTES });
  CHECK(Cleanup, test_wait_flag(&w.done, HANDOVER_TIMEOUT_MS) == 0);

  io_event_wait(q, ev_all);
  CHECK(Cleanup, io_queue_pending_bytes(q) == 0);
  rc = 0;

Cleanup:
  answer_the_rest(q, &fake, &answered);
  test_thread_join(thr);
  io_queue_destroy(q);
  return rc;

Fail:
  return 1;
}

// --- test: a barrier waits for an outstanding write on the same file ---

static int
barrier_after_write(int write_status)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  io_backend_fake_defer(&fake, 1);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){ 0 });
  CHECK(Fail, q);

  int rc = 1;
  uint64_t answered = 0;

  CHECK(Cleanup,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .file = { .generation = 7 },
                                           .nbytes = WRITE_BYTES }) == 0);
  CHECK(Cleanup,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_CLOSE,
                                           .file = { .generation = 7 } }) == 0);

  CHECK(Cleanup, wait_for_records(&fake, 1, HANDOVER_TIMEOUT_MS) == 0);
  platform_sleep_ns((int64_t)HOLD_OBSERVE_MS * 1000000LL);
  CHECK(Cleanup, io_backend_fake_record_count(&fake) == 1);
  CHECK(Cleanup, fake.records[0].op == IO_OP_WRITE);

  answer(
    q,
    &answered,
    (struct io_completion){ .seq = 1,
                            .nbytes = write_status == IO_OK ? WRITE_BYTES : 0,
                            .status = write_status });

  CHECK(Cleanup, wait_for_records(&fake, 2, HANDOVER_TIMEOUT_MS) == 0);
  CHECK(Cleanup, fake.records[1].op == IO_OP_CLOSE);
  rc = 0;

Cleanup:
  answer_the_rest(q, &fake, &answered);
  io_queue_destroy(q);
  return rc;

Fail:
  return 1;
}

static int
test_barrier_waits_for_write(void)
{
  return barrier_after_write(IO_OK);
}

// Skipping the close would leak the descriptor the failed write named.
static int
test_barrier_runs_after_failed_write(void)
{
  return barrier_after_write(IO_FAILED);
}

// --- test: a short or failed answer still returns all the room ---

static int
test_room_returns_after_short_answer(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  io_backend_fake_defer(&fake, 1);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){ 0 });
  CHECK(Fail, q);

  int rc = 1;
  uint64_t answered = 0;
  test_thread* thr = NULL;
  struct fence_wait w;

  CHECK(Cleanup,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .file = { .generation = 1 },
                                           .nbytes = WRITE_BYTES }) == 0);
  CHECK(Cleanup,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .file = { .generation = 2 },
                                           .nbytes = 2 * WRITE_BYTES }) == 0);
  CHECK(Cleanup, wait_for_records(&fake, 2, HANDOVER_TIMEOUT_MS) == 0);
  CHECK(Cleanup, fence_wait_start(&w, &thr, q, io_queue_record(q)) == 0);

  answer(
    q,
    &answered,
    (struct io_completion){ .seq = 1, .nbytes = 100, .status = IO_PARTIAL });
  answer(q,
         &answered,
         (struct io_completion){ .seq = 2, .nbytes = 0, .status = IO_FAILED });

  CHECK(Cleanup, test_wait_flag(&w.done, HANDOVER_TIMEOUT_MS) == 0);
  CHECK(Cleanup, io_queue_pending_bytes(q) == 0);

  // The same short count, this time reported by the backend as it finishes.
  io_backend_fake_defer(&fake, 0);
  io_backend_fake_set_outcome(&fake, IO_PARTIAL, 100);
  CHECK(Cleanup,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .file = { .generation = 3 },
                                           .nbytes = 4 * WRITE_BYTES }) == 0);
  io_event_wait(q, io_queue_record(q));
  CHECK(Cleanup, io_queue_pending_bytes(q) == 0);
  rc = 0;

Cleanup:
  answer_the_rest(q, &fake, &answered);
  test_thread_join(thr);
  io_queue_destroy(q);
  return rc;

Fail:
  return 1;
}

// --- test: a zero-byte write ---

static int
test_zero_byte_write(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){ 0 });
  CHECK(Fail, q);

  CHECK(Fail2,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .file = { .generation = 1 },
                                           .nbytes = 0 }) == 0);
  io_event_wait(q, io_queue_record(q));

  CHECK(Fail2, io_backend_fake_record_count(&fake) == 1);
  CHECK(Fail2, io_queue_pending_bytes(q) == 0);

  struct io_queue_stats st;
  io_queue_get_stats(q, &st);
  CHECK(Fail2, st.writes == 0);
  CHECK(Fail2, st.bytes_copied == 0);
  CHECK(Fail2, st.bytes_borrowed == 0);
  CHECK(Fail2, st.files_waiting_peak == 0);

  io_queue_destroy(q);
  return 0;

Fail2:
  io_queue_destroy(q);
Fail:
  return 1;
}

// --- test: a cancelled answer retires like any other ---

static int
test_cancelled_answer_retires(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  io_backend_fake_defer(&fake, 1);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){ 0 });
  CHECK(Fail, q);

  int rc = 1;
  uint64_t answered = 0;
  test_thread* thr = NULL;
  struct fence_wait w;

  CHECK(Cleanup,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .file = { .generation = 1 },
                                           .nbytes = WRITE_BYTES }) == 0);
  CHECK(Cleanup, wait_for_records(&fake, 1, HANDOVER_TIMEOUT_MS) == 0);
  CHECK(Cleanup, fence_wait_start(&w, &thr, q, io_queue_record(q)) == 0);

  answer(
    q,
    &answered,
    (struct io_completion){ .seq = 1, .nbytes = 0, .status = IO_CANCELLED });

  CHECK(Cleanup, test_wait_flag(&w.done, HANDOVER_TIMEOUT_MS) == 0);
  CHECK(Cleanup, io_queue_pending_bytes(q) == 0);
  rc = 0;

Cleanup:
  answer_the_rest(q, &fake, &answered);
  test_thread_join(thr);
  io_queue_destroy(q);
  return rc;

Fail:
  return 1;
}

// --- test: a fence recorded at the same time as a destroy ---

#define FENCE_DESTROY_ROUNDS 200
#define FENCE_DESTROY_JOBS 16

static int
fence_during_destroy_round(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){ 0 });
  CHECK(Fail, q);

  int rc = 1;
  test_thread* thr = NULL;
  struct fence_wait w;

  for (int i = 0; i < FENCE_DESTROY_JOBS; ++i)
    CHECK(Fail2,
          io_queue_post(q,
                        (struct io_request){ .op = IO_OP_WRITE,
                                             .file = { .generation = 1 },
                                             .nbytes = WRITE_BYTES }) == 0);

  CHECK(Fail2, fence_wait_start(&w, &thr, q, io_queue_record(q)) == 0);
  CHECK(Fail3, test_wait_flag(&w.entered, HANDOVER_TIMEOUT_MS) == 0);

  io_queue_destroy(q);
  q = NULL;

  CHECK(Fail3, test_wait_flag(&w.done, HANDOVER_TIMEOUT_MS) == 0);
  rc = 0;

Fail3:
  io_queue_destroy(q);
  test_thread_join(thr);
  return rc;

Fail2:
  io_queue_destroy(q);
Fail:
  return 1;
}

static int
test_fence_during_destroy(void)
{
  for (int round = 0; round < FENCE_DESTROY_ROUNDS; ++round)
    CHECK(Fail, fence_during_destroy_round() == 0);
  return 0;

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
    { "owned_payload_released", test_owned_payload_released },
    { "destroy_drains", test_destroy_drains },
    { "empty_queue_event", test_empty_queue_event },
    { "pending_bytes", test_pending_bytes },
    { "stats", test_stats },
    { "timing_mean_counts_finished", test_timing_mean_counts_finished },
    { "out_of_order_retirement", test_out_of_order_retirement },
    { "barrier_waits_for_write", test_barrier_waits_for_write },
    { "barrier_runs_after_failed_write", test_barrier_runs_after_failed_write },
    { "room_returns_after_short_answer", test_room_returns_after_short_answer },
    { "zero_byte_write", test_zero_byte_write },
    { "cancelled_answer_retires", test_cancelled_answer_retires },
    { "fence_during_destroy", test_fence_during_destroy },
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
