// The work the io queue is asked to do, described rather than wrapped in a
// closure, so a scheduler can reorder and gate it.
#pragma once

#include <stdint.h>

enum io_op
{
  IO_OP_NOOP = 0, // no file, no payload: a fence marker and a fault peg
  IO_OP_WRITE,    // payload to a file at an offset
  IO_OP_TRUNCATE, // barrier: set the file's size
  IO_OP_CLOSE,    // barrier: last request naming this token
};

// One open of one shard file. A generation is never reused, so a late request
// naming a closed file is refused rather than applied to whoever holds the
// descriptor now. Generation 0 means no file.
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

  // Every op other than a write carries no bytes, which the counters and the
  // timing both key on.
  const void* payload;
  uint64_t nbytes;
  uint64_t offset;

  uint64_t logical_size; // truncate only

  // Released after the request retires. NULL when the payload is borrowed.
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
