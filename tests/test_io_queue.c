#include "io_queue.h"
#include "prelude.h"

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
    io_queue_post(q, order_fn, &ctxs[i], NULL);
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
  io_queue_post(q, set_value, (void*)&val, NULL);
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
    io_queue_post(q, noop_fn, &free_count, free_counter);

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
    io_queue_post(q, increment, (void*)&count, NULL);

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
