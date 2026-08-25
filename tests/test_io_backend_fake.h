// This backend is for tests only: nothing is run, every request is recorded,
// and a test can hold, defer, refuse, or fail each one.
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
  // Claimed on the way in, published on the way out, so the two counts differ
  // only while a worker is between them.
  _Atomic uint64_t nclaimed;
  _Atomic uint64_t nrecords;

  uint64_t deferred[IO_BACKEND_FAKE_CAPACITY];
  _Atomic(const struct io_request*) deferred_requests[IO_BACKEND_FAKE_CAPACITY];
  _Atomic uint64_t nclaimed_deferred;
  _Atomic uint64_t ndeferred;

  _Atomic uint64_t inside_execute;

  _Atomic int* gate;
  _Atomic uint8_t defer;
  uint8_t outcome_chosen;
  int outcome_status;
  uint64_t outcome_nbytes;

  _Atomic uint64_t refusals_left;
  _Atomic uint64_t nrefused;

  void* dest;
  uint64_t dest_nbytes;
  uint64_t bytes_per_attempt;
  _Atomic uint64_t write_attempts;

  // These are written only at teardown, so neither is shared.
  uint64_t stops;
  uint64_t records_when_stopped;
};

void
io_backend_fake_init(struct io_backend_fake* f);

struct io_backend
io_backend_fake_as_backend(struct io_backend_fake* f);

// Hold every request inside execute until *gate is non-zero. The caller owns
// the gate; with a null gate nothing is held.
void
io_backend_fake_hold(struct io_backend_fake* f, _Atomic int* gate);

// Answer IO_SUBMITTED instead of finishing in place. Every deferred sequence
// number is kept in deferred[], to hand to io_queue_complete in any order.
void
io_backend_fake_defer(struct io_backend_fake* f, int defer);

// Report this status and count instead of IO_OK and the full request size.
void
io_backend_fake_set_outcome(struct io_backend_fake* f,
                            int status,
                            uint64_t nbytes);

// The next n requests handed over are refused, and every one after that is
// taken. Each refusal is recorded.
void
io_backend_fake_refuse(struct io_backend_fake* f, uint64_t requests);

// Every write carrying a payload is copied into the buffer at its own offset.
// A write reaching past the end is dropped.
void
io_backend_fake_write_into(struct io_backend_fake* f,
                           void* dest,
                           uint64_t nbytes);

// At most this many bytes are written per attempt, and the rest is retried.
// Zero means a write is done in one attempt.
void
io_backend_fake_short_write(struct io_backend_fake* f, uint64_t nbytes);

// The request a deferred call was handed is returned here, good until that
// request is reported finished. Null means that place is not filled in yet.
const struct io_request*
io_backend_fake_deferred_request(const struct io_backend_fake* f, uint64_t i);

uint64_t
io_backend_fake_record_count(const struct io_backend_fake* f);

uint64_t
io_backend_fake_refused_count(const struct io_backend_fake* f);

uint64_t
io_backend_fake_write_attempts(const struct io_backend_fake* f);

uint64_t
io_backend_fake_deferred_count(const struct io_backend_fake* f);

// Requests inside execute right now are counted. With deferring off and this
// at zero, the deferred list is final.
uint64_t
io_backend_fake_inside_execute(const struct io_backend_fake* f);
