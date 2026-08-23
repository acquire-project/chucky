#include "platform/platform.h"
#include "test_io_backend_fake.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr/io_queue.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

// A backend hands out a fresh generation with the index of the slot it took,
// and the queue finds a file by that index. The two only have to agree.
static struct io_file_token
file_token(uint64_t generation)
{
  return (struct io_file_token){ .generation = generation,
                                 .index = (uint32_t)(generation - 1) };
}

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
                                             .file = file_token(1),
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
                                             .file = file_token(1),
                                             .payload = b,
                                             .nbytes = sizeof(*b),
                                             .owned = b,
                                             .owned_free = release_block }) ==
            0);
  }

  // A payload is released after its request retires, outside the lock, so the
  // count is only final after the join.
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
                                             .file = file_token(1),
                                             .nbytes = nbytes }) == 0);
    posted += nbytes;
    CHECK(Fail3, io_queue_pending_bytes(q) == posted);
  }

  atomic_store(&gate, 1);
  io_event_wait(q, io_queue_record(q));
  CHECK(Fail2, io_queue_pending_bytes(q) == 0);

  // The figure is left alone by a request posted without a byte count.
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
                                   .file = file_token(file),
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
                                           .file = file_token(4),
                                           .logical_size = 4096 }) == 0);
  CHECK(Fail3,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_CLOSE,
                                           .file = file_token(5) }) == 0);
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
                                           .file = file_token(1),
                                           .nbytes = 4096 }) == 0);
  atomic_store(&gate, 1);
  io_event_wait(q, io_queue_record(q));

  // Five more are posted behind the second gate and stay unfinished.
  _Atomic int held;
  atomic_store(&held, 0);
  io_backend_fake_hold(&fake, &held);
  CHECK(Fail4, io_queue_post(q, (struct io_request){ .op = IO_OP_NOOP }) == 0);
  for (int i = 0; i < 5; ++i)
    CHECK(Fail4,
          io_queue_post(q,
                        (struct io_request){ .op = IO_OP_WRITE,
                                             .file = file_token(1),
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

// Poll until the fake has been handed n requests. Zero is returned once it
// has, -1 if the wait ran out.
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

// Poll until the backend has answered n requests with IO_SUBMITTED. Zero is
// returned once it has, -1 if the wait ran out.
static int
wait_for_deferred(const struct io_backend_fake* f, uint64_t n, int timeout_ms)
{
  int waited_ms = 0;
  while (io_backend_fake_deferred_count(f) < n) {
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

// Every exit from a deferring case has to come through here, because destroy
// is blocked until each deferred request has been answered.
static void
answer_the_rest(struct io_queue* q,
                struct io_backend_fake* f,
                uint64_t* answered)
{
  // A barrier freed by one of these answers must not be deferred again.
  io_backend_fake_defer(f, 0);

  // A request already inside execute when defer was cleared is still added to
  // the list, and leaving it unanswered hangs destroy. Once nothing is inside
  // execute the list is final, because the cleared flag is read on the way in.
  for (int waited_ms = 0; io_backend_fake_inside_execute(f) > 0; ++waited_ms) {
    if (waited_ms >= HANDOVER_TIMEOUT_MS) {
      // Reading the list now races the backend, and with a missed entry a
      // running job is left behind to hang destroy.
      log_error("io_queue test: the backend never came out of execute");
      break;
    }
    platform_sleep_ns(1000000LL);
  }

  uint64_t n = io_backend_fake_deferred_count(f);
  if (n > IO_BACKEND_FAKE_CAPACITY)
    n = IO_BACKEND_FAKE_CAPACITY;
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
// rather than a hung test.
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
                                           .file = file_token(1),
                                           .nbytes = WRITE_BYTES }) == 0);
  const struct io_event ev_first = io_queue_record(q);

  for (uint64_t i = 1; i < OUT_OF_ORDER_COUNT; ++i)
    CHECK(Cleanup,
          io_queue_post(q,
                        (struct io_request){ .op = IO_OP_WRITE,
                                             .file = file_token(i + 1),
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

  // Three are answered and the oldest is not, so the watermark has not moved.
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
                                           .file = file_token(7),
                                           .nbytes = WRITE_BYTES }) == 0);
  CHECK(Cleanup,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_CLOSE,
                                           .file = file_token(7) }) == 0);

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
                                           .file = file_token(1),
                                           .nbytes = WRITE_BYTES }) == 0);
  CHECK(Cleanup,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .file = file_token(2),
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

  // The same short count is reported by the backend this time, as it finishes.
  io_backend_fake_defer(&fake, 0);
  io_backend_fake_set_outcome(&fake, IO_PARTIAL, 100);
  CHECK(Cleanup,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .file = file_token(3),
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

// --- test: the byte ceiling holds a post back ---

struct ceiling_post
{
  struct io_queue* q;
  _Atomic int entered;
  _Atomic int done;
};

static void
ceiling_post_fn(void* arg)
{
  struct ceiling_post* c = (struct ceiling_post*)arg;
  atomic_store(&c->entered, 1);
  io_queue_post(c->q,
                (struct io_request){ .op = IO_OP_WRITE,
                                     .file = file_token(1),
                                     .nbytes = WRITE_BYTES });
  atomic_store(&c->done, 1);
}

static int
test_byte_ceiling_holds_a_post(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);

  struct io_queue* q =
    io_queue_create(io_backend_fake_as_backend(&fake),
                    (struct io_queue_limits){ .max_bytes = WRITE_BYTES });
  CHECK(Fail, q);

  int rc = 1;
  test_thread* thr = NULL;
  struct ceiling_post c = { .q = q };
  atomic_store(&c.entered, 0);
  atomic_store(&c.done, 0);

  _Atomic int gate;
  atomic_store(&gate, 0);
  io_backend_fake_hold(&fake, &gate);

  CHECK(Cleanup,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .file = file_token(1),
                                           .nbytes = WRITE_BYTES }) == 0);
  CHECK(Cleanup, io_queue_pending_bytes(q) == WRITE_BYTES);

  CHECK(Cleanup, test_thread_start(&thr, ceiling_post_fn, &c) == 0);
  CHECK(Cleanup, test_wait_flag(&c.entered, HANDOVER_TIMEOUT_MS) == 0);

  // The ceiling is already spent, so the second post cannot get through until
  // the first write retires.
  CHECK(Cleanup, test_wait_flag(&c.done, HOLD_OBSERVE_MS) == -1);
  CHECK(Cleanup, io_queue_pending_bytes(q) == WRITE_BYTES);

  atomic_store(&gate, 1);
  CHECK(Cleanup, test_wait_flag(&c.done, HANDOVER_TIMEOUT_MS) == 0);

  io_event_wait(q, io_queue_record(q));
  CHECK(Cleanup, io_queue_pending_bytes(q) == 0);
  CHECK(Cleanup, io_backend_fake_record_count(&fake) == 2);
  rc = 0;

Cleanup:
  atomic_store(&gate, 1);
  test_thread_join(thr);
  io_queue_destroy(q);
  return rc;

Fail:
  return 1;
}

static int
test_byte_ceiling_admits_an_oversize_write(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);

  struct io_queue* q =
    io_queue_create(io_backend_fake_as_backend(&fake),
                    (struct io_queue_limits){ .max_bytes = WRITE_BYTES });
  CHECK(Fail, q);

  CHECK(Fail2,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .file = file_token(1),
                                           .nbytes = WRITE_BYTES * 8 }) == 0);
  io_event_wait(q, io_queue_record(q));
  CHECK(Fail2, io_queue_pending_bytes(q) == 0);
  CHECK(Fail2, io_backend_fake_record_count(&fake) == 1);

  io_queue_destroy(q);
  return 0;

Fail2:
  io_queue_destroy(q);
Fail:
  return 1;
}

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
                                           .file = file_token(1),
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
                                           .file = file_token(1),
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
                                             .file = file_token(1),
                                             .nbytes = WRITE_BYTES }) == 0);

  CHECK(Fail2, fence_wait_start(&w, &thr, q, io_queue_record(q)) == 0);
  CHECK(Fail3, test_wait_flag(&w.entered, HANDOVER_TIMEOUT_MS) == 0);
  // The flag is set just before the wait, so it does not mean the thread is
  // inside the queue yet. Only threads already parked are drained by destroy;
  // one still on its way to the lock would be left holding a freed lock. Wait
  // until the thread is either parked or already back out.
  for (int waited_ms = 0; !io_queue_parked_threads(q); ++waited_ms) {
    if (atomic_load(&w.done))
      break;
    CHECK(Fail3, waited_ms < HANDOVER_TIMEOUT_MS);
    platform_sleep_ns(1000000LL);
  }

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

// --- test: several workers run several writes at once ---

// Zero when the backend has been handed exactly n requests and is handed no
// more while the observation window runs.
static int
holds_at_records(const struct io_backend_fake* f, uint64_t n)
{
  if (wait_for_records(f, n, HANDOVER_TIMEOUT_MS))
    return 1;
  platform_sleep_ns((int64_t)HOLD_OBSERVE_MS * 1000000LL);
  return io_backend_fake_record_count(f) != n;
}

#define BACKLOG 8

// Deferring holds every request open, so what the backend has been handed is
// what is running.
static int
in_flight_ceilings(struct io_queue_limits limits, uint64_t expect_running)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  io_backend_fake_defer(&fake, 1);

  struct io_queue* q =
    io_queue_create(io_backend_fake_as_backend(&fake), limits);
  CHECK(Fail, q);

  int rc = 1;
  uint64_t answered = 0;

  for (uint64_t i = 0; i < BACKLOG; ++i)
    CHECK(Cleanup,
          io_queue_post(q,
                        (struct io_request){ .op = IO_OP_WRITE,
                                             .file = file_token(1),
                                             .nbytes = WRITE_BYTES,
                                             .offset = i * WRITE_BYTES }) == 0);

  CHECK(Cleanup, holds_at_records(&fake, expect_running) == 0);

  // One answered frees exactly one slot, so one more is handed over.
  answer(q,
         &answered,
         (struct io_completion){ .seq = fake.deferred[0],
                                 .nbytes = WRITE_BYTES });
  CHECK(Cleanup, holds_at_records(&fake, expect_running + 1) == 0);
  rc = 0;

Cleanup:
  answer_the_rest(q, &fake, &answered);
  io_queue_destroy(q);
  return rc;

Fail:
  return 1;
}

static int
test_writes_in_flight_ceiling(void)
{
  return in_flight_ceilings((struct io_queue_limits){
                              .workers = BACKLOG,
                              .writes_in_flight = 3,
                              .writes_in_flight_per_file = BACKLOG,
                            },
                            3);
}

static int
test_per_file_ceiling(void)
{
  return in_flight_ceilings((struct io_queue_limits){
                              .workers = BACKLOG,
                              .writes_in_flight_per_file = 2,
                            },
                            2);
}

// --- test: files take turns ---

#define TURN_FILES 3
#define TURN_WRITES_PER_FILE 4

// One file's whole backlog is posted before the next file's, and there is
// room for fewer requests than one file has. Taking them in the order posted
// would run three writes to the first file and leave the others idle.
static int
test_files_take_turns(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);

  _Atomic int gate;
  atomic_store(&gate, 0);
  io_backend_fake_hold(&fake, &gate);

  struct io_queue* q = io_queue_create(
    io_backend_fake_as_backend(&fake),
    (struct io_queue_limits){
      .workers = TURN_FILES,
      .writes_in_flight = TURN_FILES,
      .writes_in_flight_per_file = TURN_WRITES_PER_FILE,
    });
  CHECK(Fail, q);

  int rc = 1;
  uint64_t answered = 0;

  io_backend_fake_defer(&fake, 1);

  // A request naming no file is always ready, so one per worker parks every
  // worker and then holds every in-flight slot. Nothing can be picked up
  // while the backlog is posted, so the order it is picked up in afterwards
  // is the scheduler's choice rather than a race with the posting.
  for (uint64_t i = 0; i < TURN_FILES; ++i)
    CHECK(Cleanup,
          io_queue_post(q, (struct io_request){ .op = IO_OP_NOOP }) == 0);
  CHECK(Cleanup, wait_for_records(&fake, TURN_FILES, HANDOVER_TIMEOUT_MS) == 0);

  for (uint64_t f = 1; f <= TURN_FILES; ++f)
    for (uint64_t i = 0; i < TURN_WRITES_PER_FILE; ++i)
      CHECK(Cleanup,
            io_queue_post(q,
                          (struct io_request){ .op = IO_OP_WRITE,
                                               .file = file_token(f),
                                               .nbytes = WRITE_BYTES,
                                               .offset = i * WRITE_BYTES }) ==
              0);

  atomic_store(&gate, 1);
  CHECK(Cleanup,
        wait_for_deferred(&fake, TURN_FILES, HANDOVER_TIMEOUT_MS) == 0);
  for (uint64_t seq = 1; seq <= TURN_FILES; ++seq)
    answer(q, &answered, (struct io_completion){ .seq = seq });

  CHECK(Cleanup, holds_at_records(&fake, 2 * TURN_FILES) == 0);

  uint64_t seen = 0;
  for (uint64_t i = TURN_FILES; i < 2 * TURN_FILES; ++i) {
    const uint64_t bit = (uint64_t)1 << fake.records[i].generation;
    CHECK(Cleanup, (seen & bit) == 0);
    seen |= bit;
  }
  rc = 0;

Cleanup:
  atomic_store(&gate, 1);
  answer_the_rest(q, &fake, &answered);
  io_queue_destroy(q);
  return rc;

Fail:
  return 1;
}

// --- test: a barrier still waits when many workers are free ---

#define BARRIER_WRITES 4

static int
test_barrier_waits_with_many_workers(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  io_backend_fake_defer(&fake, 1);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){
                                         .workers = BACKLOG,
                                         .writes_in_flight_per_file = BACKLOG,
                                       });
  CHECK(Fail, q);

  int rc = 1;
  uint64_t answered = 0;

  for (uint64_t i = 0; i < BARRIER_WRITES; ++i)
    CHECK(Cleanup,
          io_queue_post(q,
                        (struct io_request){ .op = IO_OP_WRITE,
                                             .file = file_token(1),
                                             .nbytes = WRITE_BYTES,
                                             .offset = i * WRITE_BYTES }) == 0);
  CHECK(Cleanup,
        io_queue_post(
          q, (struct io_request){ .op = IO_OP_CLOSE, .file = file_token(1) }) ==
          0);

  // Idle workers and no write ahead of the close retired, so the close waits.
  CHECK(Cleanup, holds_at_records(&fake, BARRIER_WRITES) == 0);
  for (uint64_t i = 0; i < BARRIER_WRITES; ++i)
    CHECK(Cleanup, fake.records[i].op == IO_OP_WRITE);

  for (uint64_t seq = 1; seq <= BARRIER_WRITES; ++seq)
    answer(q,
           &answered,
           (struct io_completion){ .seq = seq, .nbytes = WRITE_BYTES });

  CHECK(Cleanup, wait_for_records(&fake, BARRIER_WRITES + 1, HANDOVER_TIMEOUT_MS) == 0);
  CHECK(Cleanup, fake.records[BARRIER_WRITES].op == IO_OP_CLOSE);
  rc = 0;

Cleanup:
  answer_the_rest(q, &fake, &answered);
  io_queue_destroy(q);
  return rc;

Fail:
  return 1;
}

// --- test: a new file claiming a closing file's index ---

// A backend hands an index back when the close naming it runs, so a new file
// can name that index while the old close has not retired. Both have to
// finish, and the new file's write must not be held behind the old close.
static int
test_index_reused_before_close_retires(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  io_backend_fake_defer(&fake, 1);

  struct io_queue* q = io_queue_create(io_backend_fake_as_backend(&fake),
                                       (struct io_queue_limits){
                                         .workers = 2,
                                         .writes_in_flight_per_file = 2,
                                       });
  CHECK(Fail, q);

  int rc = 1;
  uint64_t answered = 0;
  const struct io_file_token first = { .generation = 1, .index = 0 };
  const struct io_file_token second = { .generation = 2, .index = 0 };

  CHECK(Cleanup,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .file = first,
                                           .nbytes = WRITE_BYTES }) == 0);
  CHECK(Cleanup,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_CLOSE,
                                           .file = first }) == 0);
  CHECK(Cleanup, wait_for_records(&fake, 1, HANDOVER_TIMEOUT_MS) == 0);
  answer(
    q, &answered, (struct io_completion){ .seq = 1, .nbytes = WRITE_BYTES });
  CHECK(Cleanup, wait_for_records(&fake, 2, HANDOVER_TIMEOUT_MS) == 0);
  CHECK(Cleanup, fake.records[1].op == IO_OP_CLOSE);

  // The close is running and has not retired, which is the window the
  // backend hands the index back in.
  CHECK(Cleanup,
        io_queue_post(q,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .file = second,
                                           .nbytes = WRITE_BYTES }) == 0);
  CHECK(Cleanup, wait_for_records(&fake, 3, HANDOVER_TIMEOUT_MS) == 0);
  CHECK(Cleanup, fake.records[2].generation == 2);

  answer(q, &answered, (struct io_completion){ .seq = 2 });
  answer(
    q, &answered, (struct io_completion){ .seq = 3, .nbytes = WRITE_BYTES });

  io_event_wait(q, io_queue_record(q));
  CHECK(Cleanup, io_queue_pending_bytes(q) == 0);
  rc = 0;

Cleanup:
  answer_the_rest(q, &fake, &answered);
  io_queue_destroy(q);
  return rc;

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
    { "byte_ceiling_holds_a_post", test_byte_ceiling_holds_a_post },
    { "byte_ceiling_admits_an_oversize_write",
      test_byte_ceiling_admits_an_oversize_write },
    { "zero_byte_write", test_zero_byte_write },
    { "cancelled_answer_retires", test_cancelled_answer_retires },
    { "fence_during_destroy", test_fence_during_destroy },
    { "writes_in_flight_ceiling", test_writes_in_flight_ceiling },
    { "per_file_ceiling", test_per_file_ceiling },
    { "files_take_turns", test_files_take_turns },
    { "barrier_waits_with_many_workers",
      test_barrier_waits_with_many_workers },
    { "index_reused_before_close_retires",
      test_index_reused_before_close_retires },
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
