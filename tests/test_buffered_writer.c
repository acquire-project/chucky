#include "writer.buffered.h"

#include "platform/platform.h"
#include "util/prelude.h"

#include <stdatomic.h>
#include <string.h>

struct fake_writer
{
  struct writer writer;
  struct platform_mutex* mutex;
  struct platform_cond* changed;
  int held;
  int hold_each;
  int entered;
  size_t append_bytes[128];
  size_t step;
  size_t limit;
  int terminal;
  int stall;
  int flush_error;
  int close_error;
  unsigned char output[128];
  size_t bytes;
  int appends;
  int flushes;
  int closes;
};

static struct writer_result
fake_append(struct writer* self, struct slice input)
{
  struct fake_writer* f = (struct fake_writer*)self;
  platform_mutex_lock(f->mutex);
  if (f->hold_each)
    f->held = 1;
  if (f->entered < 128)
    f->append_bytes[f->entered] =
      (const unsigned char*)input.end - (const unsigned char*)input.beg;
  ++f->entered;
  platform_cond_broadcast(f->changed);
  while (f->held)
    platform_cond_wait(f->changed, f->mutex);
  platform_mutex_unlock(f->mutex);
  ++f->appends;
  if (f->stall)
    return (struct writer_result){ 0, input };
  size_t n = (const unsigned char*)input.end - (const unsigned char*)input.beg;
  if (n > f->step)
    n = f->step;
  if (n > f->limit - f->bytes)
    n = f->limit - f->bytes;
  memcpy(f->output + f->bytes, input.beg, n);
  f->bytes += n;
  input.beg = (const unsigned char*)input.beg + n;
  return (struct writer_result){ f->bytes == f->limit ? f->terminal : 0,
                                 input };
}

static struct writer_result
fake_flush(struct writer* self)
{
  struct fake_writer* f = (struct fake_writer*)self;
  ++f->flushes;
  return f->flush_error ? writer_error() : writer_ok();
}

static struct writer_result
fake_close(struct writer* self)
{
  struct fake_writer* f = (struct fake_writer*)self;
  ++f->closes;
  return f->close_error ? writer_error() : writer_ok();
}

static void
unhold(struct fake_writer* f)
{
  platform_mutex_lock(f->mutex);
  f->hold_each = 0;
  f->held = 0;
  platform_cond_broadcast(f->changed);
  platform_mutex_unlock(f->mutex);
}

static void
wait_call(struct fake_writer* f, int count)
{
  platform_mutex_lock(f->mutex);
  while (f->entered < count)
    platform_cond_wait(f->changed, f->mutex);
  platform_mutex_unlock(f->mutex);
}

static void
release_call(struct fake_writer* f)
{
  platform_mutex_lock(f->mutex);
  f->held = 0;
  platform_cond_broadcast(f->changed);
  platform_mutex_unlock(f->mutex);
}

static void
wait_entered(struct fake_writer* f)
{
  platform_mutex_lock(f->mutex);
  while (!f->entered)
    platform_cond_wait(f->changed, f->mutex);
  platform_mutex_unlock(f->mutex);
}

static int
fake_init(struct fake_writer* f)
{
  *f = (struct fake_writer){
    .writer = { fake_append, fake_flush, fake_close },
    .mutex = platform_mutex_new(),
    .changed = platform_cond_new(),
    .step = 128,
    .limit = 128,
    .terminal = writer_error_finished,
  };
  return !f->mutex || !f->changed;
}

static void
fake_free(struct fake_writer* f)
{
  platform_cond_free(f->changed);
  platform_mutex_free(f->mutex);
}

struct producer
{
  struct writer* writer;
  struct slice input;
  struct writer_result result;
  atomic_int started;
  atomic_int done;
};

static void
produce(void* arg)
{
  struct producer* p = arg;
  atomic_store(&p->started, 1);
  p->result = writer_append(p->writer, p->input);
  atomic_store(&p->done, 1);
}

// Hold the consumer before it reads: memory reuse, FIFO, slot ownership during
// forwarding, partial acceptance, blocking at capacity, and failure wakeups.
static int
test_queue(int terminal, size_t limit)
{
  struct fake_writer f;
  if (fake_init(&f))
    return 1;
  f.held = 1;
  f.step = 3; // downstream partial progress
  if (terminal) {
    f.limit = limit;
    f.terminal = terminal;
  }
  struct buffered_writer* b =
    buffered_writer_create(&f.writer, &(struct buffered_writer_config){ 8, 2 });
  struct platform_thread* thread = NULL;
  CHECK(Fail, b);
  struct writer* w = buffered_writer_as_writer(b);
  unsigned char input[24];
  for (size_t i = 0; i < sizeof(input); ++i)
    input[i] = (unsigned char)i;
  struct slice slice = { input, input + sizeof(input) };
  struct writer_result r = writer_append(w, slice);
  CHECK(Fail, !r.error && r.rest.beg == input + 8 && r.rest.end == slice.end);
  memset(input, 0xFF, 8);
  wait_entered(&f);
  r = writer_append(w, r.rest);
  CHECK(Fail, !r.error && r.rest.beg == input + 16);
  memset(input + 8, 0xFF, 8);
  struct buffered_writer_stats stats = buffered_writer_get_stats(b);
  CHECK(Fail, stats.accepted_bytes == 16 && stats.forwarded_bytes == 0);
  CHECK(Fail, stats.pending_bytes == 16 && stats.occupied_slots == 2);

  struct producer p = { .writer = w, .input = r.rest };
  atomic_init(&p.started, 0);
  atomic_init(&p.done, 0);
  thread = platform_thread_start(produce, &p);
  CHECK(Fail, thread);
  while (!atomic_load(&p.started))
    platform_sleep_ns(100000);
  CHECK(Fail, !atomic_load(&p.done)); // no slot can be reclaimed while held
  unhold(&f);
  platform_thread_join(thread);
  thread = NULL;
  if (terminal) {
    CHECK(Fail, p.result.error == writer_error_fail);
    CHECK(Fail,
          p.result.rest.beg == input + 16 && p.result.rest.end == slice.end);
  } else {
    CHECK(Fail, !p.result.error && p.result.rest.beg == slice.end);
  }
  memset(input, 0xFF, sizeof(input));
  CHECK(Fail, writer_flush(w).error == (terminal ? writer_error_fail : 0));
  CHECK(Fail, writer_flush(w).error == (terminal ? writer_error_fail : 0));
  CHECK(Fail, writer_close(w).error == (terminal ? writer_error_fail : 0));
  CHECK(Fail, writer_close(w).error == (terminal ? writer_error_fail : 0));
  CHECK(Fail, f.flushes == 1 && f.closes == 1);
  stats = buffered_writer_get_stats(b);
  CHECK(Fail, !stats.pending_bytes && !stats.occupied_slots);
  CHECK(Fail, stats.peak_pending_bytes == 16 && stats.peak_occupied_slots == 2);
  CHECK(Fail, stats.forwarded_bytes == (terminal ? limit : 24u));
  CHECK(Fail, stats.abandoned_bytes == (terminal ? 16u - limit : 0u));
  CHECK(Fail,
        stats.accepted_bytes == stats.forwarded_bytes + stats.abandoned_bytes);
  CHECK(Fail, stats.downstream_finished == (terminal == writer_error_finished));
  CHECK(Fail, stats.failed == (terminal != 0));
  for (size_t i = 0; i < f.bytes; ++i)
    CHECK(Fail, f.output[i] == i);
  r = writer_append(w, slice);
  CHECK(Fail,
        r.error == (terminal ? writer_error_fail : writer_error_finished));
  CHECK(Fail, r.rest.beg == slice.beg && r.rest.end == slice.end);
  int error = buffered_writer_destroy(b);
  b = NULL;
  CHECK(Fail, error == (terminal != 0));
  fake_free(&f);
  return 0;
Fail:
  unhold(&f);
  if (thread)
    platform_thread_join(thread);
  buffered_writer_destroy(b);
  fake_free(&f);
  return 1;
}

static int
test_finalize(int flush_error, int close_error, int finish, int stall)
{
  struct fake_writer f;
  if (fake_init(&f))
    return 1;
  f.flush_error = flush_error;
  f.close_error = close_error;
  f.stall = stall;
  if (finish)
    f.limit = 8;
  struct buffered_writer* b =
    buffered_writer_create(&f.writer, &(struct buffered_writer_config){ 8, 1 });
  CHECK(Fail, b);
  struct writer* w = buffered_writer_as_writer(b);
  CHECK(Fail, writer_close(w).error == 0 && f.closes == 0);
  unsigned char input[8] = { 1, 2, 3 };
  CHECK(Fail, writer_append(w, (struct slice){ NULL, NULL }).error == 0);
  CHECK(Fail,
        writer_append_wait(w, (struct slice){ input, input + 8 }).error == 0);
  if (finish) {
    struct writer_result r =
      writer_append(w, (struct slice){ input, input + 8 });
    CHECK(Fail, r.error == writer_error_finished && r.rest.beg == input);
  }
  CHECK(Fail, writer_flush(w).error == (flush_error || stall));
  CHECK(Fail, writer_close(w).error == (flush_error || close_error || stall));
  CHECK(Fail, writer_close(w).error == (flush_error || close_error || stall));
  struct buffered_writer_stats stats = buffered_writer_get_stats(b);
  CHECK(Fail, stats.abandoned_bytes == (stall ? 8u : 0u));
  CHECK(Fail, f.flushes == 1 && f.closes == 1);
  int error = buffered_writer_destroy(b);
  b = NULL;
  CHECK(Fail, error == (flush_error || close_error || stall));
  // Borrowed downstream is still alive, and cleanup ran only once.
  CHECK(Fail, f.flushes == 1 && f.closes == 1);
  fake_free(&f);
  return 0;
Fail:
  buffered_writer_destroy(b);
  fake_free(&f);
  return 1;
}

// Hold each drain while the producer builds the next backlog. Vary append
// sizes and alternate buffers repeatedly: every backlog must arrive as one
// packed append, including when a drain terminates inside the combined input.
static int
test_backlog(int terminal)
{
  struct fake_writer f;
  if (fake_init(&f))
    return 1;
  f.hold_each = 1;
  if (terminal) {
    f.limit = 10; // one byte, then a prefix of the first combined drain
    f.terminal = terminal;
  }
  struct buffered_writer* b =
    buffered_writer_create(&f.writer, &(struct buffered_writer_config){ 8, 6 });
  CHECK(Fail, b);
  struct writer* w = buffered_writer_as_writer(b);
  unsigned char input[64];
  for (size_t i = 0; i < sizeof(input); ++i)
    input[i] = (unsigned char)i;
  CHECK(Fail, !writer_append(w, (struct slice){ input, input + 1 }).error);
  wait_call(&f, 1);
  const size_t sizes[][3] = { { 3, 8, 2 }, { 7, 1, 5 }, { 2, 4, 8 } };
  size_t offset = 1;
  for (int batch = 0; batch < (terminal ? 1 : 3); ++batch) {
    size_t bytes = 0;
    for (int slot = 0; slot < 3; ++slot) {
      size_t n = sizes[batch][slot];
      CHECK(
        Fail,
        !writer_append(w, (struct slice){ input + offset, input + offset + n })
           .error);
      memset(input + offset, 0xFF, n);
      offset += n;
      bytes += n;
    }
    release_call(&f);
    wait_call(&f, batch + 2);
    CHECK(Fail, f.append_bytes[batch + 1] == bytes);
    struct buffered_writer_stats stats = buffered_writer_get_stats(b);
    CHECK(Fail, stats.occupied_slots == 3 && stats.pending_bytes == bytes);
  }
  if (terminal) {
    // Accepted during the combined downstream call: these must also be
    // abandoned when it returns a terminal result, without another append.
    CHECK(
      Fail,
      !writer_append(w, (struct slice){ input + offset, input + offset + 4 })
         .error);
    offset += 4;
  }
  memset(input, 0xFF, sizeof(input));
  unhold(&f);
  CHECK(Fail, writer_flush(w).error == (terminal != 0));
  CHECK(Fail, f.appends == (terminal ? 2 : 4));
  struct buffered_writer_stats stats = buffered_writer_get_stats(b);
  CHECK(Fail, stats.completed_batches == (terminal ? 2u : 4u));
  CHECK(Fail, stats.completed_slots == (terminal ? 4u : 10u));
  CHECK(Fail, stats.max_batch_bytes == (terminal ? 13u : 14u));
  CHECK(Fail, stats.forwarded_bytes == (terminal ? 10u : offset));
  CHECK(Fail, stats.abandoned_bytes == (terminal ? offset - 10 : 0));
  CHECK(Fail, !stats.occupied_slots && !stats.pending_bytes);
  CHECK(Fail, stats.downstream_finished == (terminal == writer_error_finished));
  for (size_t i = 0; i < f.bytes; ++i)
    CHECK(Fail, f.output[i] == i);
  int error = buffered_writer_destroy(b);
  b = NULL;
  CHECK(Fail, error == (terminal != 0));
  fake_free(&f);
  return 0;
Fail:
  unhold(&f);
  buffered_writer_destroy(b);
  fake_free(&f);
  return 1;
}

static int
test_destroy_and_config(void)
{
  struct fake_writer f;
  if (fake_init(&f))
    return 1;
  struct buffered_writer_config cfg = { 4, 2 };
  struct buffered_writer* b = NULL;
  CHECK(Fail, !buffered_writer_create(NULL, &cfg));
  CHECK(Fail, !buffered_writer_create(&f.writer, NULL));
  CHECK(Fail,
        !buffered_writer_create(&f.writer,
                                &(struct buffered_writer_config){ 0, 2 }));
  CHECK(Fail,
        !buffered_writer_create(&f.writer,
                                &(struct buffered_writer_config){ 4, 0 }));
  CHECK(Fail,
        !buffered_writer_create(
          &f.writer, &(struct buffered_writer_config){ 2, SIZE_MAX }));
  CHECK(Fail, buffered_writer_destroy(NULL) == 0);
  b = buffered_writer_create(&f.writer, &cfg);
  CHECK(Fail, b);
  unsigned char input[16] = { 1, 2, 3, 4 };
  CHECK(Fail,
        !writer_append_wait(buffered_writer_as_writer(b),
                            (struct slice){ input, input + 16 })
           .error);
  memset(input, 0xFF, sizeof(input));
  int error = buffered_writer_destroy(b);
  b = NULL;
  CHECK(Fail, !error && f.bytes == 16 && f.flushes == 1 && f.closes == 1);
  CHECK(Fail, f.output[0] == 1 && f.output[3] == 4 && f.output[15] == 0);
  fake_free(&f);
  return 0;
Fail:
  buffered_writer_destroy(b);
  fake_free(&f);
  return 1;
}

int
main(void)
{
  int failed = test_destroy_and_config();
  failed += test_queue(0, 0);
  failed += test_queue(writer_error_fail, 3);
  failed += test_queue(writer_error_finished, 3);
  failed += test_queue(writer_error_finished, 8);
  failed += test_backlog(0);
  failed += test_backlog(writer_error_fail);
  failed += test_backlog(writer_error_finished);
  failed += test_finalize(0, 0, 0, 0);
  failed += test_finalize(1, 0, 0, 0);
  failed += test_finalize(0, 1, 0, 0);
  failed += test_finalize(0, 0, 1, 0);
  failed += test_finalize(0, 0, 0, 1);
  return failed != 0;
}
