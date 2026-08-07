#ifndef TEST_SHARD_SINK_H
#define TEST_SHARD_SINK_H

#include "writer.h"

#include <stddef.h>
#include <stdint.h>

#define TEST_SHARD_SINK_MAX_SHARDS 64
#define TEST_SHARD_SINK_MAX_LEVELS 8

struct test_shard_writer
{
  struct shard_writer base;
  struct test_shard_sink* sink; // back-pointer for finalize
  uint8_t* buf;
  size_t capacity;
  size_t size;
  int finalized;
  uint8_t level;
  uint64_t shard_index;
};

struct test_shard_sink
{
  struct shard_sink base;
  struct test_shard_writer writers[TEST_SHARD_SINK_MAX_LEVELS]
                                  [TEST_SHARD_SINK_MAX_SHARDS];
  struct shard_writer discard_writer; // for levels beyond num_levels
  int num_levels;
  int num_shards_per_level[TEST_SHARD_SINK_MAX_LEVELS];
  size_t per_shard_capacity;
  size_t shard_alignment; // 0 = no alignment; non-zero exposes carry-over path
  int open_count;
  int finalize_count;
  int update_append_count;
  // Counts write_direct vs write calls so tests can verify the deliver path
  // takes the zero-copy direct path on aligned runs.
  int write_count;
  int write_direct_count;
  // Once this many bytes have been accepted, every later write fails. 0 =
  // never. Lets a test put the failure after some data has landed rather than
  // on the first write.
  size_t fail_writes_after_bytes;
  size_t written_bytes; // accepted only
};

// Initialize a multi-level sink.
void
test_sink_init_multi(struct test_shard_sink* s,
                     int num_levels,
                     const int* num_shards_per_level,
                     size_t per_shard_capacity);

// Convenience: single-level sink with num_shards shards.
void
test_sink_init(struct test_shard_sink* s,
               int num_shards,
               size_t per_shard_capacity);

void
test_sink_free(struct test_shard_sink* s);

#endif // TEST_SHARD_SINK_H
