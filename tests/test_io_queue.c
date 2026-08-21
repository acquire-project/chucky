#include "test_io_backend_fake.h"
#include "util/prelude.h"
#include "zarr/io_queue.h"

#include <stdatomic.h>
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
