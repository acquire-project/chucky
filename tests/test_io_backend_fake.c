#include "test_io_backend_fake.h"
#include "platform/platform.h"

#include <string.h>

static int
take_a_refusal(struct io_backend_fake* f)
{
  uint64_t left = atomic_load(&f->refusals_left);
  while (left > 0 &&
         !atomic_compare_exchange_weak(&f->refusals_left, &left, left - 1)) {
  }
  if (left == 0)
    return 0;
  atomic_fetch_add(&f->nrefused, 1);
  return 1;
}

static void
copy_the_write(struct io_backend_fake* f, const struct io_request* req)
{
  struct io_request rest = io_write_remaining(req, 0);
  while (rest.nbytes > 0) {
    uint64_t taken = rest.nbytes;
    if (f->bytes_per_attempt && taken > f->bytes_per_attempt)
      taken = f->bytes_per_attempt;
    if (rest.offset <= f->dest_nbytes && taken <= f->dest_nbytes - rest.offset)
      memcpy((char*)f->dest + rest.offset, rest.payload, (size_t)taken);
    atomic_fetch_add(&f->write_attempts, 1);
    rest = io_write_remaining(&rest, taken);
  }
}

static int
fake_execute(void* ctx,
             const struct io_request* req,
             uint64_t seq,
             struct io_completion* out)
{
  struct io_backend_fake* f = (struct io_backend_fake*)ctx;

  atomic_fetch_add(&f->inside_execute, 1);

  // Each worker claims its own slot, then the count is raised, so a test that
  // polls the count reads records that are already filled in.
  const uint64_t n = atomic_fetch_add(&f->nclaimed, 1);
  if (n < IO_BACKEND_FAKE_CAPACITY)
    f->records[n] = (struct io_backend_fake_record){
      .seq = seq,
      .nbytes = req->nbytes,
      .generation = req->file.generation,
      .op = req->op,
    };
  atomic_fetch_add(&f->nrecords, 1);

  // The hold is first, so a request can be both held and refused.
  while (f->gate && atomic_load(f->gate) == 0)
    platform_sleep_ns(1000000LL);

  int dispatch = IO_BUSY;
  if (!take_a_refusal(f)) {
    if (f->dest && req->payload && req->op == IO_OP_WRITE)
      copy_the_write(f, req);

    out->seq = seq;
    out->nbytes = f->outcome_chosen ? f->outcome_nbytes : req->nbytes;
    out->status = f->outcome_chosen ? f->outcome_status : IO_OK;

    dispatch = IO_DONE;
    if (atomic_load(&f->defer)) {
      const uint64_t d = atomic_fetch_add(&f->nclaimed_deferred, 1);
      if (d < IO_BACKEND_FAKE_CAPACITY) {
        f->deferred[d] = seq;
        atomic_store(&f->deferred_requests[d], req);
      }
      atomic_fetch_add(&f->ndeferred, 1);
      dispatch = IO_SUBMITTED;
    }
  }

  atomic_fetch_sub(&f->inside_execute, 1);
  return dispatch;
}

static void
fake_stop(void* ctx)
{
  struct io_backend_fake* f = (struct io_backend_fake*)ctx;
  f->stops++;
  f->records_when_stopped = atomic_load(&f->nrecords);
}

void
io_backend_fake_init(struct io_backend_fake* f)
{
  memset(f, 0, sizeof(*f));
  atomic_init(&f->nclaimed, 0);
  atomic_init(&f->nrecords, 0);
  atomic_init(&f->nclaimed_deferred, 0);
  atomic_init(&f->ndeferred, 0);
  atomic_init(&f->inside_execute, 0);
  atomic_init(&f->refusals_left, 0);
  atomic_init(&f->nrefused, 0);
  atomic_init(&f->write_attempts, 0);
}

struct io_backend
io_backend_fake_as_backend(struct io_backend_fake* f)
{
  return (struct io_backend){ .ctx = f,
                              .execute = fake_execute,
                              .stop = fake_stop };
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

void
io_backend_fake_refuse(struct io_backend_fake* f, uint64_t requests)
{
  atomic_store(&f->refusals_left, requests);
}

void
io_backend_fake_write_into(struct io_backend_fake* f,
                           void* dest,
                           uint64_t nbytes)
{
  f->dest = dest;
  f->dest_nbytes = nbytes;
}

void
io_backend_fake_short_write(struct io_backend_fake* f, uint64_t nbytes)
{
  f->bytes_per_attempt = nbytes;
}

const struct io_request*
io_backend_fake_deferred_request(const struct io_backend_fake* f, uint64_t i)
{
  if (i >= IO_BACKEND_FAKE_CAPACITY)
    return NULL;
  return atomic_load(&f->deferred_requests[i]);
}

uint64_t
io_backend_fake_record_count(const struct io_backend_fake* f)
{
  return atomic_load(&f->nrecords);
}

uint64_t
io_backend_fake_refused_count(const struct io_backend_fake* f)
{
  return atomic_load(&f->nrefused);
}

uint64_t
io_backend_fake_write_attempts(const struct io_backend_fake* f)
{
  return atomic_load(&f->write_attempts);
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
