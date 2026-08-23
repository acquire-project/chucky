#include "test_io_backend_fake.h"
#include "platform/platform.h"

#include <string.h>

static int
fake_execute(void* ctx,
             const struct io_request* req,
             uint64_t seq,
             struct io_completion* out)
{
  struct io_backend_fake* f = (struct io_backend_fake*)ctx;

  atomic_fetch_add(&f->inside_execute, 1);

  // The count is raised last, so a test that polls it then reads a record
  // that is already there. With one worker, only this thread picks a slot.
  const uint64_t n = atomic_load(&f->nrecords);
  if (n < IO_BACKEND_FAKE_CAPACITY)
    f->records[n] = (struct io_backend_fake_record){
      .seq = seq,
      .nbytes = req->nbytes,
      .generation = req->file.generation,
      .op = req->op,
    };
  atomic_fetch_add(&f->nrecords, 1);

  while (f->gate && atomic_load(f->gate) == 0)
    platform_sleep_ns(1000000LL);

  out->seq = seq;
  out->nbytes = f->outcome_chosen ? f->outcome_nbytes : req->nbytes;
  out->status = f->outcome_chosen ? f->outcome_status : IO_OK;

  int dispatch = IO_DONE;
  if (atomic_load(&f->defer)) {
    const uint64_t d = atomic_load(&f->ndeferred);
    if (d < IO_BACKEND_FAKE_CAPACITY)
      f->deferred[d] = seq;
    atomic_fetch_add(&f->ndeferred, 1);
    dispatch = IO_SUBMITTED;
  }

  atomic_fetch_sub(&f->inside_execute, 1);
  return dispatch;
}

void
io_backend_fake_init(struct io_backend_fake* f)
{
  memset(f, 0, sizeof(*f));
  atomic_init(&f->nrecords, 0);
  atomic_init(&f->ndeferred, 0);
  atomic_init(&f->inside_execute, 0);
}

struct io_backend
io_backend_fake_as_backend(struct io_backend_fake* f)
{
  return (struct io_backend){ .ctx = f, .execute = fake_execute };
}

void
io_backend_fake_hold(struct io_backend_fake* f, _Atomic int* gate)
{
  f->gate = gate;
}

void
io_backend_fake_defer(struct io_backend_fake* f, int defer)
{
  atomic_store(&f->defer, (uint8_t)(defer != 0));
}

void
io_backend_fake_set_outcome(struct io_backend_fake* f,
                            int status,
                            uint64_t nbytes)
{
  f->outcome_chosen = 1;
  f->outcome_status = status;
  f->outcome_nbytes = nbytes;
}

uint64_t
io_backend_fake_record_count(const struct io_backend_fake* f)
{
  return atomic_load(&f->nrecords);
}

uint64_t
io_backend_fake_deferred_count(const struct io_backend_fake* f)
{
  return atomic_load(&f->ndeferred);
}

uint64_t
io_backend_fake_inside_execute(const struct io_backend_fake* f)
{
  return atomic_load(&f->inside_execute);
}
