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
  writer_error_finished = 2, // stream complete: total_element_limit reached
};

struct writer_result
{
  int error;         // writer_error_code; 0 = ok, 1 = fail, 2 = finished
  struct slice rest; // unconsumed input (empty on success for append)
};

struct writer
{
  struct writer_result (*append)(struct writer* self, struct slice data);
  // explicit sync; stream remains appendable. Returns when all writes prior
  // to the call are durable.
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

  // Returns non-zero if any async IO has failed. NULL = no async IO.
  int (*has_error)(const struct shard_sink* self);

  // Returns bytes queued but not yet retired on this sink. NULL = treated as 0.
  size_t (*pending_bytes)(const struct shard_sink* self);

  // Required write alignment in bytes (e.g. page size for O_DIRECT).
  // NULL or returns 0 = no alignment constraint.
  size_t (*required_shard_alignment)(const struct shard_sink* self);
};

size_t
shard_sink_pending_bytes(const struct shard_sink* s);

size_t
shard_sink_required_shard_alignment(const struct shard_sink* s);

// Drain pending async IO on this sink across all levels. Returns non-zero
// if the sink reports an error after drain.
int
shard_sink_drain(struct shard_sink* s, int nlod);

// Two-phase drain: record fences for all levels, then wait. Lets callers
// fan out the record phase across many sinks before blocking on waits.
// `evs` must point to at least `nlod` io_event slots.
void
shard_sink_drain_record(struct shard_sink* s, int nlod, struct io_event* evs);

int
shard_sink_drain_wait(struct shard_sink* s,
                      int nlod,
                      const struct io_event* evs);

// Drain N sinks together. Returns the number of sinks reporting errors.
int
shard_sink_drain_many(struct shard_sink** sinks, const int* nlods, int n);

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
