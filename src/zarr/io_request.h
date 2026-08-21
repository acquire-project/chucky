// Work for the io queue is described here rather than wrapped in a closure,
// so it can be reordered and gated.
#pragma once

#include <stdint.h>

enum io_op
{
  IO_OP_NOOP = 0, // no file, no payload: a fence marker and a test hook
  IO_OP_WRITE,    // payload to a file at an offset
  IO_OP_TRUNCATE, // barrier: set the file's size
  IO_OP_CLOSE,    // barrier: last request naming this token
};

// A token names one open of one shard file. A generation is never reused, so
// a late request naming a closed file is refused rather than applied to
// whoever holds the descriptor now. Generation 0 means no file.
struct io_file_token
{
  uint64_t generation;
  uint32_t index;
  uint32_t reserved;
};

struct io_request
{
  uint8_t op;
  uint8_t borrowed; // payload memory owned elsewhere
  uint8_t reserved[6];

  struct io_file_token file;

  // Only a write has bytes; the counters and the timing both depend on that.
  const void* payload;
  uint64_t nbytes;
  uint64_t offset;

  uint64_t logical_size; // truncate only

  // The owned buffer is released after the request retires; a borrowed
  // payload has none.
  void* owned;
  void (*owned_free)(void*);
};

enum io_status
{
  IO_OK = 0,
  IO_PARTIAL,
  IO_FAILED,
  IO_CANCELLED,
};

struct io_completion
{
  uint64_t seq;
  uint64_t nbytes; // written; never more than the request asked for
  int status;
};
