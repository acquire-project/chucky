// A test-only backend: nothing is run, every request is recorded, and a test
// can hold, defer, or fail what the io queue hands over.
#pragma once

#include "zarr/io_backend.h"

#include <stdatomic.h>
#include <stdint.h>

#define IO_BACKEND_FAKE_CAPACITY 256

struct io_backend_fake_record
{
  uint64_t seq;
  uint64_t nbytes;
  uint64_t generation;
  uint8_t op;
};

struct io_backend_fake
{
  struct io_backend_fake_record records[IO_BACKEND_FAKE_CAPACITY];
  _Atomic uint64_t nrecords;

  uint64_t deferred[IO_BACKEND_FAKE_CAPACITY];
  _Atomic uint64_t ndeferred;

  _Atomic uint64_t inside_execute;

  _Atomic int* gate;
  _Atomic uint8_t defer;
  uint8_t outcome_chosen;
  int outcome_status;
  uint64_t outcome_nbytes;
};

void
io_backend_fake_init(struct io_backend_fake* f);

struct io_backend
io_backend_fake_as_backend(struct io_backend_fake* f);

// Hold every request inside execute until *gate is non-zero. The test owns the
// gate; with a null gate nothing is held.
void
io_backend_fake_hold(struct io_backend_fake* f, _Atomic int* gate);

// Answer IO_SUBMITTED instead of finishing in place. Every deferred sequence
// number is kept in deferred[], to hand to io_queue_complete in any order.
void
io_backend_fake_defer(struct io_backend_fake* f, int defer);

// Report this instead of "all of it, fine".
void
io_backend_fake_set_outcome(struct io_backend_fake* f,
                            int status,
                            uint64_t nbytes);

uint64_t
io_backend_fake_record_count(const struct io_backend_fake* f);

uint64_t
io_backend_fake_deferred_count(const struct io_backend_fake* f);

// Requests inside execute right now. With deferring off and this at zero, the
// deferred list is final.
uint64_t
io_backend_fake_inside_execute(const struct io_backend_fake* f);
