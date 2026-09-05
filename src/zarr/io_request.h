// Work is described here rather than wrapped in a closure so the scheduler can
// preserve file ordering.
#pragma once

#include <stdint.h>

enum io_op
{
  IO_OP_NOOP = 0, // no file or payload; used by fault-injection tests
  IO_OP_OPEN,     // barrier: create or truncate a file
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

  struct io_file_token file;

  const char* path;

  const void* payload;
  uint64_t nbytes;
  uint64_t offset;

  uint64_t logical_size; // truncate only

  // The owned allocation is released before the request is reported complete.
  void* owned;
  void (*owned_free)(void*);

  void* finished_ctx;
  void (*finished)(void* ctx);
};
