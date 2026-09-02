#pragma once

#include "zarr/io_backend.h"

#include <stdatomic.h>
#include <stdint.h>

#define IO_BACKEND_FAKE_CAPACITY 256

struct io_backend_fake_record
{
  uint64_t nbytes;
  uint64_t offset;
  uint64_t logical_size;
  uint64_t generation;
  uint8_t op;
  _Atomic int ready;
};

struct io_backend_fake
{
  struct io_backend_fake_record records[IO_BACKEND_FAKE_CAPACITY];
  _Atomic uint64_t started;
  _Atomic uint64_t active;
  _Atomic uint64_t active_peak;
  _Atomic int* gate;
  void* dest;
  uint64_t dest_nbytes;
};

void
io_backend_fake_init(struct io_backend_fake* fake);

struct io_backend
io_backend_fake_as_backend(struct io_backend_fake* fake);

void
io_backend_fake_hold(struct io_backend_fake* fake, _Atomic int* gate);

void
io_backend_fake_write_into(struct io_backend_fake* fake,
                           void* dest,
                           uint64_t nbytes);

uint64_t
io_backend_fake_started(const struct io_backend_fake* fake);

uint64_t
io_backend_fake_active(const struct io_backend_fake* fake);

uint64_t
io_backend_fake_active_peak(const struct io_backend_fake* fake);
