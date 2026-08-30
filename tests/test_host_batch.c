#include "stream/types.aggregate.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

#include <stdlib.h>
#include <string.h>

static int
test_span_planning(void)
{
  struct batch_aggregate_layout layout = {
    .nlod = 2,
    .page_size = 64,
    .total_batch_covering = 13,
    .total_data_bytes = 576,
    .lods = {
      { .covering_count = 4,
        .n_active = 3,
        .batch_covering_offset = 0,
        .data_segment_offset = 0,
        .data_segment_bytes = 512 },
      { .covering_count = 1,
        .n_active = 1,
        .batch_covering_offset = 12,
        .data_segment_offset = 512,
        .data_segment_bytes = 64 },
    },
  };
  uint32_t active[2] = { 3, 1 };
  size_t offsets[15] = { 0 };
  size_t sizes[15] = { 0 };
  offsets[11] = 313;
  sizes[11] = 10;
  offsets[13] = 515;

  struct d2h_transfer_span spans[2];
  size_t count = 0;
  CHECK(Fail,
        d2h_plan_legacy_spans(
          &layout, 2, active, NULL, NULL, 1, spans, 2, &count) == 0);
  CHECK(Fail, count == 1);
  CHECK(Fail, spans[0].bytes == 576);

  CHECK(Fail,
        d2h_plan_legacy_spans(
          &layout, 2, active, offsets, sizes, 0, spans, 2, &count) == 0);
  CHECK(Fail, count == 2);
  CHECK(Fail, spans[0].device_offset == 0 && spans[0].bytes == 323);
  CHECK(Fail, spans[1].device_offset == 512 && spans[1].bytes == 3);

  CHECK(Fail,
        d2h_plan_legacy_spans(
          &layout, 2, active, offsets, sizes, 0, spans, 1, &count) != 0);
  CHECK(Fail, count == 0);
  return 0;

Fail:
  return 1;
}

static int
test_host_batch_runs(void)
{
  struct active_shard lod0_active[2] = { 0 };
  struct active_shard lod1_active[1] = { 0 };
  lod0_active[0].tail_bytes = 5;
  lod0_active[1].tail_bytes = 7;
  lod1_active[0].tail_bytes = 3;
  struct shard_state lod0 = {
    .epoch_in_shard = 1,
    .shard_epoch = 3,
    .shard_inner_count = 2,
    .chunks_per_shard_inner = 2,
    .chunks_per_shard_append = 2,
    .shards = lod0_active,
  };
  struct shard_state lod1 = {
    .epoch_in_shard = 0,
    .shard_epoch = 9,
    .shard_inner_count = 1,
    .chunks_per_shard_inner = 1,
    .chunks_per_shard_append = 4,
    .shards = lod1_active,
  };
  struct shard_state* shards[2] = { &lod0, &lod1 };
  struct aggregate_layout per_lod[2] = {
    { .page_size = 64, .shard_capacity = 256 },
    { .page_size = 64, .shard_capacity = 64 },
  };
  struct batch_aggregate_layout layout = {
    .nlod = 2,
    .page_size = 64,
    .total_batch_covering = 13,
    .total_data_bytes = 576,
    .lods = {
      { .covering_count = 4,
        .n_active = 3,
        .batch_covering_offset = 0,
        .data_segment_offset = 0,
        .data_segment_bytes = 512 },
      { .covering_count = 1,
        .n_active = 1,
        .batch_covering_offset = 12,
        .data_segment_offset = 512,
        .data_segment_bytes = 64 },
    },
  };
  uint32_t active[2] = { 3, 1 };
  size_t offsets[15] = { 0 };
  size_t sizes[15] = { 0 };

  // LOD 0: two shards, three active epochs, two chunks each.  Each shard's
  // first run carries its committed tail; the second starts a fresh shard.
  for (size_t j = 0; j < 6; ++j) {
    offsets[j] = 5 + j * 10;
    sizes[j] = 10;
    offsets[6 + j] = 256 + 7 + j * 10;
    sizes[6 + j] = 10;
  }
  // LOD 1 has a zero-sized covering entry but still carries a prior tail.
  offsets[13] = 512 + 3;
  sizes[13] = 0;

  uint8_t aggregate[576] = { 0 };
  struct host_batch host = { 0 };
  CHECK(Fail,
        host_batch_build_legacy(&host,
                                aggregate,
                                offsets,
                                sizes,
                                &layout,
                                per_lod,
                                shards,
                                active,
                                2,
                                aggregate) == 0);
  CHECK(Fail, host.run_count == 5);
  CHECK(Fail, host.transfer.logical_payload_bytes == 120);

  CHECK(Fail, host.runs[0].flat_shard == 6);
  CHECK(Fail, host.runs[0].tail_bytes == 5);
  CHECK(Fail, host.runs[0].active_count == 1);
  CHECK(Fail, host.runs[0].finalizes);
  CHECK(Fail, host.runs[1].flat_shard == 7);
  CHECK(Fail, host.runs[1].tail_bytes == 7);
  CHECK(Fail, host.runs[1].ends_generation_run);
  CHECK(Fail, host.runs[2].flat_shard == 8);
  CHECK(Fail, host.runs[2].tail_bytes == 0);
  CHECK(Fail, host.runs[2].active_count == 2);
  CHECK(Fail, host.runs[4].level == 1);
  CHECK(Fail, host.runs[4].flat_shard == 9);
  CHECK(Fail, host.runs[4].tail_bytes == 3);
  CHECK(Fail, host.runs[4].payload_bytes == 0);
  CHECK(Fail, !host.runs[4].finalizes);

  // Overflow in a run's logical extent is rejected rather than wrapped.
  sizes[0] = SIZE_MAX;
  sizes[1] = 1;
  CHECK(Fail,
        host_batch_build_legacy(&host,
                                aggregate,
                                offsets,
                                sizes,
                                &layout,
                                per_lod,
                                shards,
                                active,
                                2,
                                aggregate) != 0);

  host_batch_destroy(&host);
  return 0;

Fail:
  host_batch_destroy(&host);
  return 1;
}

int
main(void)
{
  return test_span_planning() || test_host_batch_runs();
}
