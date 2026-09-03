#pragma once

#include <stddef.h>
#include <stdint.h>

struct host_output_group;

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
  writer_error_finished = 2, // stream complete: flushed, or limit reached
};

struct writer_result
{
  int error;         // writer_error_code; 0 = ok, 1 = fail, 2 = finished
  struct slice rest; // unconsumed input (empty on success for append)
};

struct writer
{
  struct writer_result (*append)(struct writer* self, struct slice data);
  // Finalizes the stream: writes out everything appended so far, including the
  // chunk the append cursor stopped partway through, and stops taking input.
  // Idempotent. A later append consumes nothing and reports `finished`. It
  // returns once those writes have landed, so a write that failed is reported
  // here.
  //
  // Finalizing is what makes the partial chunk readable: it is padded out and
  // its shard is closed. Taking more input afterwards would have to start past
  // that padding and past the shard slots the close left empty, which puts
  // later data at append positions the caller never asked for.
  struct writer_result (*flush)(struct writer* self);

  // Optional: publishes the append extent and lets the sink write its own
  // metadata. Returns whether that succeeded, so this is where the array
  // becomes readable. Idempotent, and destroying the stream runs it if the
  // caller did not — so the sink has to outlive the stream. NULL when a writer
  // has nothing to publish.
  struct writer_result (*close)(struct writer* self);
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
  // The writer retains group for every request that still points into this
  // output and completes each retain when that request retires.
  int (*write_from_output)(struct shard_writer* self,
                           uint64_t offset,
                           const void* beg,
                           const void* end,
                           struct host_output_group* group);
  // Optional: set the shard's size to nbytes before anything is written. On
  // some filesystems a write that extends a file takes the file's lock for
  // itself, so this is what lets a shard's writes run together. No space is
  // reserved, and truncate trims the shard back. NULL = no-op.
  int (*presize)(struct shard_writer* self, uint64_t nbytes);
  // Optional: truncate the shard's persistent storage to logical_size bytes.
  // Used after O_DIRECT writes that overshoot the logical end (page-aligned)
  // so the shard file's on-disk size matches the index's expectations.
  // NULL = no-op (object stores set size implicitly).
  int (*truncate)(struct shard_writer* self, uint64_t logical_size);
  int (*finalize)(struct shard_writer* self); // shard complete, close/flush
};

struct shard_sink
{
  // Open/get a writer for the given flat shard index.
  struct shard_writer* (*open)(struct shard_sink* self,
                               uint8_t level,
                               uint64_t shard_index);

  // Optional: update append dim extents in metadata (e.g. zarr.json shape).
  // Called periodically during streaming and from the writer's close.
  // append_sizes has n_append elements (sizes for dims 0..n_append-1).
  // NULL means no-op (non-zarr sinks can ignore).
  int (*update_append)(struct shard_sink* self,
                       uint8_t level,
                       uint8_t n_append,
                       const uint64_t* append_sizes);

  // IO fence for backpressure. NULL = no async IO.
  struct io_event (*record_fence)(struct shard_sink* self);
  void (*wait_fence)(struct shard_sink* self, struct io_event ev);

  // Optional: flush sink-level state (e.g. dirty metadata).
  // Called from the writer's close, after outstanding shard IO has settled.
  // NULL = no-op.
  int (*flush)(struct shard_sink* self);

  // Returns non-zero if any async IO has failed. NULL = no async IO.
  int (*has_error)(const struct shard_sink* self);

  // Returns an upper bound on bytes accepted but not yet written; a 0 is not
  // proof that nothing is outstanding. NULL = treated as 0.
  uint64_t (*pending_bytes)(const struct shard_sink* self);

  // Required write alignment in bytes (e.g. page size for O_DIRECT).
  // NULL or returns 0 = no alignment constraint.
  size_t (*required_shard_alignment)(const struct shard_sink* self);
};

uint64_t
shard_sink_pending_bytes(const struct shard_sink* s);

size_t
shard_sink_required_shard_alignment(const struct shard_sink* s);

// Drain pending async IO on this sink. Returns non-zero if the sink
// reports an error after drain.
int
shard_sink_drain(struct shard_sink* s);

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

// Dispatch to the writer's close method. Ok when the writer has none.
struct writer_result
writer_close(struct writer* w);

// Append data to a writer, retrying with exponential back-off on stall.
struct writer_result
writer_append_wait(struct writer* w, struct slice data);
