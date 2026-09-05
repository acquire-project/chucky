#include "test_io_backend_fake.h"
#include "test_platform.h"

#include "log/log.h"
#include "platform/platform.h"
#include "util/prelude.h"
#include "zarr/io_scheduler.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define WAIT_MS 2000

static int
wait_for_started(const struct io_backend_fake* fake,
                 uint64_t count,
                 int timeout_ms)
{
  for (int elapsed = 0; elapsed < timeout_ms; ++elapsed) {
    if (io_backend_fake_started(fake) >= count) {
      int ready = 1;
      for (uint64_t i = 0; i < count; ++i)
        ready &= atomic_load(&fake->records[i].ready);
      if (ready)
        return 0;
    }
    platform_sleep_ns(1000000LL);
  }
  return 1;
}

static int
wait_for_parked(const struct io_scheduler* scheduler,
                uint64_t count,
                int timeout_ms)
{
  for (int elapsed = 0; elapsed < timeout_ms; ++elapsed) {
    if (io_scheduler_parked_threads(scheduler) >= count)
      return 0;
    platform_sleep_ns(1000000LL);
  }
  return 1;
}

static int
test_ordering_and_payload(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  uint8_t output[12] = { 0 };
  io_backend_fake_write_into(&fake, output, sizeof(output));

  struct io_scheduler* scheduler =
    io_scheduler_create(io_backend_fake_as_backend(&fake),
                        (struct io_scheduler_limits){ .workers = 1 });
  CHECK(Fail, scheduler);

  static const uint8_t first[] = { 1, 2, 3, 4 };
  static const uint8_t second[] = { 5, 6, 7, 8 };
  static const uint8_t third[] = { 9, 10, 11, 12 };
  const uint8_t* payloads[] = { first, second, third };
  for (uint64_t i = 0; i < 3; ++i)
    CHECK(FailQueue,
          io_scheduler_post(scheduler,
                            (struct io_request){
                              .op = IO_OP_WRITE,
                              .payload = payloads[i],
                              .nbytes = 4,
                              .offset = i * 4,
                            }) == 0);

  io_event_wait(scheduler, io_scheduler_record(scheduler));
  CHECK(FailQueue, wait_for_started(&fake, 3, WAIT_MS) == 0);
  CHECK(FailQueue, memcmp(output, first, 4) == 0);
  CHECK(FailQueue, memcmp(output + 4, second, 4) == 0);
  CHECK(FailQueue, memcmp(output + 8, third, 4) == 0);
  io_scheduler_destroy(scheduler);
  return 0;

FailQueue:
  io_scheduler_destroy(scheduler);
Fail:
  return 1;
}

static int
test_empty_event(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  struct io_scheduler* scheduler = io_scheduler_create(
    io_backend_fake_as_backend(&fake), (struct io_scheduler_limits){ 0 });
  CHECK(Fail, scheduler);
  io_event_wait(scheduler, io_scheduler_record(scheduler));
  CHECK(FailQueue, io_backend_fake_started(&fake) == 0);
  io_scheduler_destroy(scheduler);
  return 0;

FailQueue:
  io_scheduler_destroy(scheduler);
Fail:
  return 1;
}

struct owned_payload
{
  _Atomic int* released;
  uint8_t bytes[8];
};

static void
release_owned(void* data)
{
  struct owned_payload* payload = (struct owned_payload*)data;
  atomic_fetch_add(payload->released, 1);
  free(payload);
}

struct finished_capture
{
  _Atomic int called;
};

static void
capture_finished(void* ctx)
{
  struct finished_capture* capture = (struct finished_capture*)ctx;
  atomic_store(&capture->called, 1);
}

static int
test_owned_payload_and_callback(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  _Atomic int released = 0;
  struct finished_capture capture = { 0 };
  struct owned_payload* payload =
    (struct owned_payload*)calloc(1, sizeof(*payload));
  CHECK(Fail, payload);
  payload->released = &released;

  struct io_scheduler* scheduler = io_scheduler_create(
    io_backend_fake_as_backend(&fake), (struct io_scheduler_limits){ 0 });
  CHECK(FailPayload, scheduler);
  CHECK(FailQueue,
        io_scheduler_post(scheduler,
                          (struct io_request){
                            .op = IO_OP_WRITE,
                            .payload = payload->bytes,
                            .nbytes = sizeof(payload->bytes),
                            .owned = payload,
                            .owned_free = release_owned,
                            .finished_ctx = &capture,
                            .finished = capture_finished,
                          }) == 0);
  payload = NULL;
  io_scheduler_destroy(scheduler);
  scheduler = NULL;

  CHECK(FailPayload, atomic_load(&released) == 1);
  CHECK(FailPayload, atomic_load(&capture.called) == 1);
  return 0;

FailQueue:
  io_scheduler_destroy(scheduler);
FailPayload:
  free(payload);
Fail:
  return 1;
}

struct post_call
{
  struct io_scheduler* scheduler;
  struct io_request request;
  _Atomic int entered;
  _Atomic int done;
  int result;
};

static void
post_main(void* arg)
{
  struct post_call* call = (struct post_call*)arg;
  atomic_store(&call->entered, 1);
  call->result = io_scheduler_post(call->scheduler, call->request);
  atomic_store(&call->done, 1);
}

static int
test_byte_ceiling(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  _Atomic int gate = 0;
  io_backend_fake_hold(&fake, &gate);
  test_thread* poster = NULL;
  struct io_scheduler* scheduler = io_scheduler_create(
    io_backend_fake_as_backend(&fake),
    (struct io_scheduler_limits){ .max_bytes = 64, .workers = 1 });
  CHECK(Fail, scheduler);

  CHECK(Cleanup,
        io_scheduler_post(
          scheduler, (struct io_request){ .op = IO_OP_WRITE, .nbytes = 64 }) ==
          0);
  CHECK(Cleanup, wait_for_started(&fake, 1, WAIT_MS) == 0);
  CHECK(Cleanup, io_scheduler_pending_bytes(scheduler) == 64);

  struct post_call call = {
    .scheduler = scheduler,
    .request = { .op = IO_OP_WRITE, .nbytes = 64 },
  };
  CHECK(Cleanup, test_thread_start(&poster, post_main, &call) == 0);
  CHECK(Cleanup, test_wait_flag(&call.entered, WAIT_MS) == 0);
  CHECK(Cleanup, wait_for_parked(scheduler, 1, WAIT_MS) == 0);
  CHECK(Cleanup, atomic_load(&call.done) == 0);

  atomic_store(&gate, 1);
  CHECK(Cleanup, test_wait_flag(&call.done, WAIT_MS) == 0);
  CHECK(Cleanup, call.result == 0);
  CHECK(Cleanup, test_thread_join(poster) == 0);
  poster = NULL;
  io_event_wait(scheduler, io_scheduler_record(scheduler));
  CHECK(Cleanup, io_scheduler_pending_bytes(scheduler) == 0);

  atomic_store(&gate, 0);
  CHECK(Cleanup,
        io_scheduler_post(
          scheduler, (struct io_request){ .op = IO_OP_WRITE, .nbytes = 128 }) ==
          0);
  CHECK(Cleanup, io_scheduler_pending_bytes(scheduler) == 128);
  atomic_store(&gate, 1);
  io_event_wait(scheduler, io_scheduler_record(scheduler));
  CHECK(Cleanup, io_scheduler_pending_bytes(scheduler) == 0);

  io_scheduler_destroy(scheduler);
  return 0;

Cleanup:
  atomic_store(&gate, 1);
  if (poster)
    test_thread_join(poster);
  io_scheduler_destroy(scheduler);
Fail:
  return 1;
}

static struct io_request
file_write(uint64_t generation, uint32_t index)
{
  return (struct io_request){
    .op = IO_OP_WRITE,
    .file = { .generation = generation, .index = index },
    .nbytes = 1,
  };
}

static int
test_per_file_ceiling(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  _Atomic int gate = 0;
  io_backend_fake_hold(&fake, &gate);
  struct io_scheduler* scheduler =
    io_scheduler_create(io_backend_fake_as_backend(&fake),
                        (struct io_scheduler_limits){
                          .workers = 4,
                          .max_in_flight_per_file = 2,
                        });
  CHECK(Fail, scheduler);

  for (int i = 0; i < 4; ++i)
    CHECK(Cleanup, io_scheduler_post(scheduler, file_write(1, 0)) == 0);
  CHECK(Cleanup, wait_for_started(&fake, 2, WAIT_MS) == 0);
  platform_sleep_ns(20000000LL);
  CHECK(Cleanup, io_backend_fake_started(&fake) == 2);
  CHECK(Cleanup, io_backend_fake_active(&fake) == 2);

  atomic_store(&gate, 1);
  io_event_wait(scheduler, io_scheduler_record(scheduler));
  CHECK(Cleanup, io_backend_fake_started(&fake) == 4);
  CHECK(Cleanup, io_backend_fake_active_peak(&fake) == 2);
  io_scheduler_destroy(scheduler);
  return 0;

Cleanup:
  atomic_store(&gate, 1);
  io_scheduler_destroy(scheduler);
Fail:
  return 1;
}

static int
test_files_take_turns(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  _Atomic int gate = 0;
  io_backend_fake_hold(&fake, &gate);
  struct io_scheduler* scheduler =
    io_scheduler_create(io_backend_fake_as_backend(&fake),
                        (struct io_scheduler_limits){
                          .workers = 1,
                          .max_in_flight_per_file = 1,
                        });
  CHECK(Fail, scheduler);

  CHECK(Cleanup, io_scheduler_post(scheduler, file_write(1, 0)) == 0);
  CHECK(Cleanup, wait_for_started(&fake, 1, WAIT_MS) == 0);
  CHECK(Cleanup, io_scheduler_post(scheduler, file_write(1, 0)) == 0);
  CHECK(Cleanup, io_scheduler_post(scheduler, file_write(2, 1)) == 0);
  atomic_store(&gate, 1);
  io_event_wait(scheduler, io_scheduler_record(scheduler));
  CHECK(Cleanup, wait_for_started(&fake, 3, WAIT_MS) == 0);
  CHECK(Cleanup, fake.records[0].generation == 1);
  CHECK(Cleanup, fake.records[1].generation == 2);
  CHECK(Cleanup, fake.records[2].generation == 1);

  io_scheduler_destroy(scheduler);
  return 0;

Cleanup:
  atomic_store(&gate, 1);
  io_scheduler_destroy(scheduler);
Fail:
  return 1;
}

static int
test_file_barrier_sequence(const uint8_t* ops, uint32_t count)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  _Atomic int gate = 0;
  io_backend_fake_hold(&fake, &gate);
  struct io_scheduler* scheduler =
    io_scheduler_create(io_backend_fake_as_backend(&fake),
                        (struct io_scheduler_limits){
                          .workers = 3,
                          .max_in_flight_per_file = 3,
                        });
  CHECK(Fail, scheduler);

  const struct io_file_token file = { .generation = 1, .index = 0 };
  for (uint32_t i = 0; i < count; ++i) {
    CHECK(Cleanup,
          io_scheduler_post(
            scheduler, (struct io_request){ .op = ops[i], .file = file }) == 0);
    if (i == 0)
      CHECK(Cleanup, wait_for_started(&fake, 1, WAIT_MS) == 0);
  }
  platform_sleep_ns(20000000LL);
  CHECK(Cleanup, io_backend_fake_started(&fake) == 1);

  atomic_store(&gate, 1);
  io_event_wait(scheduler, io_scheduler_record(scheduler));
  CHECK(Cleanup, io_backend_fake_started(&fake) == count);
  CHECK(Cleanup, io_backend_fake_active_peak(&fake) == 1);
  for (uint32_t i = 0; i < count; ++i)
    CHECK(Cleanup, fake.records[i].op == ops[i]);

  io_scheduler_destroy(scheduler);
  return 0;

Cleanup:
  atomic_store(&gate, 1);
  io_scheduler_destroy(scheduler);
Fail:
  return 1;
}

static int
test_file_barriers(void)
{
  int err = 0;
  err |=
    test_file_barrier_sequence((const uint8_t[]){ IO_OP_OPEN, IO_OP_WRITE }, 2);
  err |= test_file_barrier_sequence(
    (const uint8_t[]){ IO_OP_TRUNCATE, IO_OP_WRITE }, 2);
  err |= test_file_barrier_sequence(
    (const uint8_t[]){ IO_OP_WRITE, IO_OP_TRUNCATE, IO_OP_CLOSE }, 3);
  err |= test_file_barrier_sequence(
    (const uint8_t[]){ IO_OP_WRITE, IO_OP_CLOSE }, 2);
  return err;
}

static int
test_file_opens_overlap(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  _Atomic int gate = 0;
  io_backend_fake_hold(&fake, &gate);
  struct io_scheduler* scheduler =
    io_scheduler_create(io_backend_fake_as_backend(&fake),
                        (struct io_scheduler_limits){
                          .workers = 4,
                          .max_in_flight_per_file = 4,
                        });
  CHECK(Fail, scheduler);

  for (uint32_t i = 0; i < 2; ++i) {
    const struct io_file_token file = { .generation = i + 1, .index = i };
    CHECK(Cleanup,
          io_scheduler_post(scheduler,
                            (struct io_request){
                              .op = IO_OP_OPEN,
                              .file = file,
                            }) == 0);
    CHECK(Cleanup, io_scheduler_post(scheduler, file_write(i + 1, i)) == 0);
  }

  CHECK(Cleanup, wait_for_started(&fake, 2, WAIT_MS) == 0);
  platform_sleep_ns(20000000LL);
  CHECK(Cleanup, io_backend_fake_started(&fake) == 2);
  CHECK(Cleanup, io_backend_fake_active(&fake) == 2);
  CHECK(Cleanup, fake.records[0].op == IO_OP_OPEN);
  CHECK(Cleanup, fake.records[1].op == IO_OP_OPEN);

  atomic_store(&gate, 1);
  io_event_wait(scheduler, io_scheduler_record(scheduler));
  CHECK(Cleanup, io_backend_fake_started(&fake) == 4);
  io_scheduler_destroy(scheduler);
  return 0;

Cleanup:
  atomic_store(&gate, 1);
  io_scheduler_destroy(scheduler);
Fail:
  return 1;
}

static int
test_file_generation_reuse(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  _Atomic int gate = 0;
  io_backend_fake_hold(&fake, &gate);
  struct io_scheduler* scheduler =
    io_scheduler_create(io_backend_fake_as_backend(&fake),
                        (struct io_scheduler_limits){ .workers = 1 });
  CHECK(Fail, scheduler);

  CHECK(Cleanup, io_scheduler_post(scheduler, file_write(1, 0)) == 0);
  CHECK(Cleanup, wait_for_started(&fake, 1, WAIT_MS) == 0);
  CHECK(Cleanup,
        io_scheduler_post(scheduler,
                          (struct io_request){
                            .op = IO_OP_CLOSE,
                            .file = { .generation = 1, .index = 0 },
                          }) == 0);
  CHECK(Cleanup, io_scheduler_post(scheduler, file_write(2, 0)) != 0);
  atomic_store(&gate, 1);
  io_event_wait(scheduler, io_scheduler_record(scheduler));
  CHECK(Cleanup, wait_for_started(&fake, 2, WAIT_MS) == 0);
  CHECK(Cleanup, fake.records[0].generation == 1);
  CHECK(Cleanup, fake.records[1].generation == 1);

  atomic_store(&gate, 0);
  CHECK(Cleanup,
        io_scheduler_post(scheduler,
                          (struct io_request){
                            .op = IO_OP_CLOSE,
                            .file = { .generation = 2, .index = 0 },
                          }) == 0);
  CHECK(Cleanup, wait_for_started(&fake, 3, WAIT_MS) == 0);
  CHECK(Cleanup, io_scheduler_post(scheduler, file_write(3, 0)) == 0);
  atomic_store(&gate, 1);
  io_event_wait(scheduler, io_scheduler_record(scheduler));
  CHECK(Cleanup, wait_for_started(&fake, 4, WAIT_MS) == 0);
  CHECK(Cleanup, fake.records[2].generation == 2);
  CHECK(Cleanup, fake.records[3].generation == 3);

  io_scheduler_destroy(scheduler);
  return 0;

Cleanup:
  atomic_store(&gate, 1);
  io_scheduler_destroy(scheduler);
Fail:
  return 1;
}

struct wait_call
{
  struct io_scheduler* scheduler;
  struct io_event event;
  _Atomic int done;
};

static void
wait_main(void* arg)
{
  struct wait_call* call = (struct wait_call*)arg;
  io_event_wait(call->scheduler, call->event);
  atomic_store(&call->done, 1);
}

struct destroy_call
{
  struct io_scheduler* scheduler;
  _Atomic int done;
};

static void
destroy_main(void* arg)
{
  struct destroy_call* call = (struct destroy_call*)arg;
  io_scheduler_destroy(call->scheduler);
  atomic_store(&call->done, 1);
}

static int
test_destroy_releases_waiters_and_drains(void)
{
  struct io_backend_fake fake;
  io_backend_fake_init(&fake);
  _Atomic int gate = 0;
  io_backend_fake_hold(&fake, &gate);
  test_thread* poster_thread = NULL;
  test_thread* waiter_thread = NULL;
  test_thread* destroy_thread = NULL;
  struct io_scheduler* scheduler = io_scheduler_create(
    io_backend_fake_as_backend(&fake),
    (struct io_scheduler_limits){ .max_requests = 1, .workers = 1 });
  CHECK(Fail, scheduler);

  CHECK(Cleanup,
        io_scheduler_post(
          scheduler, (struct io_request){ .op = IO_OP_WRITE, .nbytes = 1 }) ==
          0);
  CHECK(Cleanup, wait_for_started(&fake, 1, WAIT_MS) == 0);

  struct post_call poster = {
    .scheduler = scheduler,
    .request = { .op = IO_OP_WRITE, .nbytes = 1 },
  };
  struct wait_call waiter = {
    .scheduler = scheduler,
    .event = io_scheduler_record(scheduler),
  };
  CHECK(Cleanup, test_thread_start(&poster_thread, post_main, &poster) == 0);
  CHECK(Cleanup, test_thread_start(&waiter_thread, wait_main, &waiter) == 0);
  CHECK(Cleanup, wait_for_parked(scheduler, 2, WAIT_MS) == 0);

  struct destroy_call destroy = { .scheduler = scheduler };
  CHECK(Cleanup,
        test_thread_start(&destroy_thread, destroy_main, &destroy) == 0);
  scheduler = NULL;
  CHECK(CleanupThreads, test_wait_flag(&poster.done, WAIT_MS) == 0);
  CHECK(CleanupThreads, poster.result != 0);
  CHECK(CleanupThreads, test_wait_flag(&waiter.done, WAIT_MS) == 0);
  atomic_store(&gate, 1);
  CHECK(CleanupThreads, test_wait_flag(&destroy.done, WAIT_MS) == 0);
  CHECK(CleanupThreads, io_backend_fake_started(&fake) == 1);

  test_thread_join(poster_thread);
  test_thread_join(waiter_thread);
  test_thread_join(destroy_thread);
  return 0;

CleanupThreads:
  atomic_store(&gate, 1);
  if (poster_thread)
    test_thread_join(poster_thread);
  if (waiter_thread)
    test_thread_join(waiter_thread);
  if (destroy_thread)
    test_thread_join(destroy_thread);
  return 1;

Cleanup:
  atomic_store(&gate, 1);
  if (poster_thread)
    test_thread_join(poster_thread);
  if (waiter_thread)
    test_thread_join(waiter_thread);
  io_scheduler_destroy(scheduler);
Fail:
  return 1;
}

int
main(void)
{
  int result = 0;
  struct
  {
    const char* name;
    int (*run)(void);
  } tests[] = {
    { "ordering_and_payload", test_ordering_and_payload },
    { "empty_event", test_empty_event },
    { "owned_payload_and_callback", test_owned_payload_and_callback },
    { "byte_ceiling", test_byte_ceiling },
    { "per_file_ceiling", test_per_file_ceiling },
    { "files_take_turns", test_files_take_turns },
    { "file_barriers", test_file_barriers },
    { "file_opens_overlap", test_file_opens_overlap },
    { "file_generation_reuse", test_file_generation_reuse },
    { "destroy_releases_waiters_and_drains",
      test_destroy_releases_waiters_and_drains },
  };

  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
    const int failed = tests[i].run();
    if (failed)
      log_error("FAIL: %s", tests[i].name);
    else
      log_info("PASS: %s", tests[i].name);
    result |= failed;
  }
  return result;
}
