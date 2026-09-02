#include "test_io_backend_fake.h"

#include "platform/platform.h"

#include <string.h>

static void
fake_execute(void* ctx, const struct io_request* req)
{
  struct io_backend_fake* fake = (struct io_backend_fake*)ctx;
  const uint64_t active = atomic_fetch_add(&fake->active, 1) + 1;
  uint64_t peak = atomic_load(&fake->active_peak);
  while (active > peak &&
         !atomic_compare_exchange_weak(&fake->active_peak, &peak, active)) {
  }

  const uint64_t index = atomic_fetch_add(&fake->started, 1);
  if (index < IO_BACKEND_FAKE_CAPACITY) {
    struct io_backend_fake_record* record = &fake->records[index];
    record->nbytes = req->nbytes;
    record->offset = req->offset;
    record->logical_size = req->logical_size;
    record->generation = req->file.generation;
    record->op = req->op;
    atomic_store(&record->ready, 1);
  }

  while (fake->gate && atomic_load(fake->gate) == 0)
    platform_sleep_ns(1000000LL);

  if (req->op == IO_OP_WRITE && req->payload && fake->dest &&
      req->offset <= fake->dest_nbytes &&
      req->nbytes <= fake->dest_nbytes - req->offset)
    memcpy((char*)fake->dest + req->offset, req->payload, req->nbytes);

  atomic_fetch_sub(&fake->active, 1);
}

void
io_backend_fake_init(struct io_backend_fake* fake)
{
  memset(fake, 0, sizeof(*fake));
}

struct io_backend
io_backend_fake_as_backend(struct io_backend_fake* fake)
{
  return (struct io_backend){ .ctx = fake, .execute = fake_execute };
}

void
io_backend_fake_hold(struct io_backend_fake* fake, _Atomic int* gate)
{
  fake->gate = gate;
}

void
io_backend_fake_write_into(struct io_backend_fake* fake,
                           void* dest,
                           uint64_t nbytes)
{
  fake->dest = dest;
  fake->dest_nbytes = nbytes;
}

uint64_t
io_backend_fake_started(const struct io_backend_fake* fake)
{
  return atomic_load(&fake->started);
}

uint64_t
io_backend_fake_active(const struct io_backend_fake* fake)
{
  return atomic_load(&fake->active);
}

uint64_t
io_backend_fake_active_peak(const struct io_backend_fake* fake)
{
  return atomic_load(&fake->active_peak);
}
