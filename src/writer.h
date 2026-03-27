#pragma once

#include <stddef.h>
#include <stdint.h>

struct io_event
{
  uint64_t seq;
};

struct slice
{
  const void* beg;
  const void* end;
};

enum writer_error_code
{
  writer_error_ok = 0,
  writer_error_fail = 1,
  writer_error_finished = 2, // bounded append dims capacity reached, data flushed
};

struct writer_result
{
  int error;         // writer_error_code; 0 = ok, 1 = fail, 2 = finished
  struct slice rest; // unconsumed input (empty on success for append)
};

struct writer
{
  struct writer_result (*append)(struct writer* self, struct slice data);
  struct writer_result (*flush)(struct writer* self);
};

struct shard_writer
{
  int (*write)(struct shard_writer* self,
               uint64_t offset, // byte offset within the shard
               const void* beg,
               const void* end);
  // Zero-copy write: caller guarantees buffer lifetime until io_event.
  // NULL = fall back to write (copy-based).
  int (*write_direct)(struct shard_writer* self,
                      uint64_t offset,
                      const void* beg,
                      const void* end);
  int (*finalize)(struct shard_writer* self); // shard complete, close/flush
};

struct shard_sink
{
  // Open/get a writer for the given flat shard index.
  struct shard_writer* (*open)(struct shard_sink* self,
                               uint8_t level,
                               uint64_t shard_index);

  // Optional: update append dim extents in metadata (e.g. zarr.json shape).
  // Called periodically during streaming and at final flush.
  // append_sizes has n_append elements (sizes for dims 0..n_append-1).
  // NULL means no-op (non-zarr sinks can ignore).
  int (*update_append)(struct shard_sink* self,
                       uint8_t level,
                       uint8_t n_append,
                       const uint64_t* append_sizes);

  // IO fence for backpressure. NULL = no async IO.
  struct io_event (*record_fence)(struct shard_sink* self, uint8_t level);
  void (*wait_fence)(struct shard_sink* self,
                     uint8_t level,
                     struct io_event ev);
};

struct writer_result
writer_ok(void);

struct writer_result
writer_error(void);

struct writer_result
writer_error_at(const void* beg, const void* end);

struct writer_result
writer_finished_at(const void* beg, const void* end);

// Dispatch to the writer's append method.
struct writer_result
writer_append(struct writer* w, struct slice data);

// Dispatch to the writer's flush method.
struct writer_result
writer_flush(struct writer* w);

// Append data to a writer, retrying with exponential back-off on stall.
struct writer_result
writer_append_wait(struct writer* w, struct slice data);
