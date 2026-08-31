#include "platform/platform.h"
#include "stream/types.aggregate.h"
#include "test_shard_sink.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"
#include "zarr/shard_drainer.h"

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

static int
test_compact_layout_fixed_index(void)
{
  struct aggregate_layout per_lod[1] = {
    {
      .lifted_rank = 1,
      .lifted_shape = { 4 },
      .lifted_strides = { 1 },
      .chunks_per_epoch = 3,
      .covering_count = 4,
      .max_comp_chunk_bytes = 7,
      .cps_inner = 2,
      .num_shards = 2,
      .active_count_max = 2,
      .chunks_per_shard_append = 4,
    },
  };
  uint32_t active[1] = { 2 };
  struct batch_aggregate_layout layout;
  CHECK(Fail,
        batch_aggregate_layout_init_compact(&layout, per_lod, active, 1) == 0);
  CHECK(Fail, layout.total_batch_chunks == 6);
  CHECK(Fail, layout.total_batch_covering == 8);
  CHECK(Fail, layout.total_data_bytes == 42);
  CHECK(Fail, layout.page_size == 0);

  size_t offsets[9] = { 0 };
  size_t sizes[9] = { 0 };
  CHECK(Fail,
        aggregate_fixed_host_index(&layout, per_lod, 7, offsets, sizes) == 0);
  const size_t expected_sizes[9] = { 7, 7, 7, 7, 7, 0, 7, 0, 0 };
  size_t cursor = 0;
  for (size_t i = 0; i < 9; ++i) {
    CHECK(Fail, sizes[i] == expected_sizes[i]);
    CHECK(Fail, offsets[i] == cursor);
    cursor += expected_sizes[i];
  }
  CHECK(Fail, cursor == 42);
  return 0;

Fail:
  return 1;
}

static int
test_compact_run_planning(void)
{
  const size_t page = 64;
  uint8_t tail0[64] = { 1, 2, 3, 4, 5 };
  uint8_t tail1[64] = { 11, 12, 13, 14, 15, 16, 17 };
  uint8_t tail2[64] = { 21, 22, 23 };
  struct active_shard lod0_active[2] = {
    { .tail_buf = tail0, .tail_bytes = 5 },
    { .tail_buf = tail1, .tail_bytes = 7 },
  };
  struct active_shard lod1_active[1] = {
    { .tail_buf = tail2, .tail_bytes = 3 },
  };
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
  struct active_shard lod2_active[1] = { 0 };
  struct shard_state lod2 = {
    .epoch_in_shard = 0,
    .shard_epoch = 0,
    .shard_inner_count = 1,
    .chunks_per_shard_inner = 1,
    .chunks_per_shard_append = 2,
    .shards = lod2_active,
  };
  struct shard_state* shards[3] = { &lod0, &lod1, &lod2 };
  struct aggregate_layout per_lod[3] = {
    { .chunks_per_epoch = 4,
      .covering_count = 4,
      .max_comp_chunk_bytes = 16,
      .cps_inner = 2,
      .num_shards = 2,
      .active_count_max = 3,
      .page_size = page,
      .chunks_per_shard_append = 2 },
    { .chunks_per_epoch = 1,
      .covering_count = 1,
      .max_comp_chunk_bytes = 16,
      .cps_inner = 1,
      .num_shards = 1,
      .active_count_max = 1,
      .page_size = page,
      .chunks_per_shard_append = 4 },
    { .chunks_per_epoch = 1,
      .covering_count = 1,
      .max_comp_chunk_bytes = 16,
      .cps_inner = 1,
      .num_shards = 1,
      .active_count_max = 0,
      .page_size = page,
      .chunks_per_shard_append = 2 },
  };
  uint32_t active[3] = { 3, 1, 0 };
  struct batch_aggregate_layout layout;
  CHECK(Fail,
        batch_aggregate_layout_init_compact(&layout, per_lod, active, 3) == 0);

  size_t offsets[16] = { 0 };
  size_t sizes[16] = { 0 };
  size_t source_cursor = 0;
  for (size_t i = 0; i < 16; ++i) {
    offsets[i] = source_cursor;
    // LOD 0 occupies indices 0..11; index 12 is its sentinel. LOD 1's
    // covering entry (13) and sentinel (14) are deliberately zero-sized.
    if (i < 12) {
      sizes[i] = 10;
      source_cursor += 10;
    }
  }
  CHECK(Fail, source_cursor == 120);

  size_t host_capacity = 0;
  size_t max_runs = 0;
  CHECK(Fail,
        host_batch_compact_capacity(per_lod,
                                    active,
                                    3,
                                    HOST_DELIVERY_FIXED_TAIL,
                                    page,
                                    &host_capacity,
                                    &max_runs) == 0);
  CHECK(Fail, max_runs >= 5);
  uint8_t* host_data = (uint8_t*)platform_aligned_alloc(page, host_capacity);
  CHECK(Fail, host_data);
  memset(host_data, 0, host_capacity);
  struct d2h_transfer_span* spans =
    (struct d2h_transfer_span*)calloc(max_runs, sizeof(*spans));
  CHECK(Fail, spans);
  struct host_batch host = { 0 };
  size_t span_count = 0;
  CHECK(Fail,
        host_batch_build_compact(&host,
                                 host_data,
                                 host_capacity,
                                 offsets,
                                 sizes,
                                 &layout,
                                 per_lod,
                                 shards,
                                 active,
                                 3,
                                 HOST_DELIVERY_FIXED_TAIL,
                                 page,
                                 spans,
                                 max_runs,
                                 &span_count,
                                 host_data) == 0);
  CHECK(Fail, host.run_count == 5);
  CHECK(Fail, span_count == 4);
  CHECK(Fail, host.transfer.logical_payload_bytes == 120);
  CHECK(Fail, host.runs[0].flat_shard == 6 && host.runs[0].finalizes);
  CHECK(Fail, host.runs[1].flat_shard == 7 && host.runs[1].finalizes);
  CHECK(Fail, host.runs[2].flat_shard == 8 && host.runs[2].finalizes);
  CHECK(Fail, host.runs[4].level == 1 && host.runs[4].payload_bytes == 0);
  CHECK(Fail, host.runs[4].tail_bytes == 3 && !host.runs[4].finalizes);
  CHECK(Fail, memcmp(host.runs[0].data, tail0, 5) == 0);
  CHECK(Fail, memcmp(host.runs[1].data, tail1, 7) == 0);
  CHECK(Fail, memcmp(host.runs[4].data, tail2, 3) == 0);
  for (size_t i = 0; i < host.run_count; ++i)
    CHECK(Fail, (uintptr_t)host.runs[i].data % page == 0);
  CHECK(Fail, spans[0].device_offset == 0 && spans[0].host_offset == 5);
  CHECK(Fail, spans[1].device_offset == 60);
  CHECK(Fail, spans[2].device_offset == 20);
  CHECK(Fail, spans[3].device_offset == 80);

  // Capacity and span bounds reject before any D2H can be submitted.
  CHECK(Fail,
        host_batch_build_compact(&host,
                                 host_data,
                                 1,
                                 offsets,
                                 sizes,
                                 &layout,
                                 per_lod,
                                 shards,
                                 active,
                                 3,
                                 HOST_DELIVERY_FIXED_TAIL,
                                 page,
                                 spans,
                                 max_runs,
                                 &span_count,
                                 host_data) != 0);

  host_batch_destroy(&host);
  free(spans);
  platform_aligned_free(host_data);
  return 0;

Fail:
  return 1;
}

static int
test_compact_extent_edges(void)
{
  const size_t page = 64;
  struct active_shard active_shard[1] = { 0 };
  struct shard_state shard = {
    .epoch_in_shard = 0,
    .shard_epoch = 0,
    .shard_inner_count = 1,
    .chunks_per_shard_inner = 1,
    .chunks_per_shard_append = 1,
    .shards = active_shard,
  };
  struct shard_state* shards[1] = { &shard };
  struct aggregate_layout per_lod[1] = {
    { .chunks_per_epoch = 1,
      .covering_count = 1,
      .max_comp_chunk_bytes = 130,
      .cps_inner = 1,
      .num_shards = 1,
      .active_count_max = 3,
      .page_size = page,
      .chunks_per_shard_append = 1 },
  };
  uint32_t active[1] = { 3 };
  struct batch_aggregate_layout layout;
  CHECK(Fail,
        batch_aggregate_layout_init_compact(&layout, per_lod, active, 1) == 0);
  // Empty, exact-page, and multi-page physical runs.
  size_t sizes[4] = { 0, 64, 130, 0 };
  size_t offsets[4] = { 0, 0, 64, 194 };
  size_t capacity = 0;
  size_t run_capacity = 0;
  CHECK(Fail,
        host_batch_compact_capacity(per_lod,
                                    active,
                                    1,
                                    HOST_DELIVERY_INDEXED_PADDED,
                                    page,
                                    &capacity,
                                    &run_capacity) == 0);
  uint8_t* data = (uint8_t*)platform_aligned_alloc(page, capacity);
  struct d2h_transfer_span spans[3] = { 0 };
  struct host_batch host = { 0 };
  size_t span_count = 0;
  CHECK(Fail, data);
  CHECK(Fail,
        host_batch_build_compact(&host,
                                 data,
                                 capacity,
                                 offsets,
                                 sizes,
                                 &layout,
                                 per_lod,
                                 shards,
                                 active,
                                 1,
                                 HOST_DELIVERY_INDEXED_PADDED,
                                 page,
                                 spans,
                                 3,
                                 &span_count,
                                 data) == 0);
  CHECK(Fail, host.run_count == 3 && span_count == 2);
  CHECK(Fail, host.runs[0].payload_bytes == 0 && host.runs[0].finalizes);
  CHECK(Fail, host.runs[1].payload_bytes == page && host.runs[1].finalizes);
  CHECK(Fail, host.runs[2].payload_bytes == 130 && host.runs[2].finalizes);
  CHECK(Fail, spans[0].bytes == page && spans[1].bytes == 130);

  // A batch that would cross the uint64 generation boundary is rejected
  // during host planning, before any D2H span can be submitted.
  shard.shard_epoch = UINT64_MAX;
  CHECK(Fail,
        host_batch_build_compact(&host,
                                 data,
                                 capacity,
                                 offsets,
                                 sizes,
                                 &layout,
                                 per_lod,
                                 shards,
                                 active,
                                 1,
                                 HOST_DELIVERY_INDEXED_PADDED,
                                 page,
                                 spans,
                                 3,
                                 &span_count,
                                 data) != 0);
  shard.shard_epoch = 0;

  host_batch_destroy(&host);
  platform_aligned_free(data);

  // Overflow is rejected by shared-slot capacity planning.
  per_lod[0].max_comp_chunk_bytes = SIZE_MAX;
  CHECK(Fail,
        host_batch_compact_capacity(per_lod,
                                    active,
                                    1,
                                    HOST_DELIVERY_INDEXED_PADDED,
                                    page,
                                    &capacity,
                                    &run_capacity) != 0);
  return 0;

Fail:
  return 1;
}

static void
copy_planned_spans(uint8_t* host,
                   const uint8_t* device,
                   const struct d2h_transfer_span* spans,
                   size_t count)
{
  for (size_t i = 0; i < count; ++i)
    memcpy(host + spans[i].host_offset,
           device + spans[i].device_offset,
           spans[i].bytes);
}

static int
test_compact_host_tail_delivery(void)
{
  const size_t page = 64;
  uint8_t* footer = NULL;
  uint8_t* host_data = NULL;
  struct host_batch batch = { 0 };
  struct test_shard_sink sink;
  memset(&sink, 0, sizeof(sink));

  uint8_t tail_buf[64] = { 0 };
  uint64_t index[4];
  memset(index, 0xFF, sizeof(index));
  footer = (uint8_t*)platform_aligned_alloc(page, 128);
  CHECK(Fail, footer);

  struct active_shard active = {
    .index = index,
    .tail_buf = tail_buf,
    .footer_buf = footer,
  };
  struct shard_state shard = {
    .epoch_in_shard = 0,
    .shard_epoch = 0,
    .shard_inner_count = 1,
    .chunks_per_shard_inner = 1,
    .chunks_per_shard_total = 2,
    .chunks_per_shard_append = 2,
    .shards = &active,
    .footer_capacity = 128,
  };
  struct shard_state* shards[1] = { &shard };
  struct aggregate_layout per_lod[1] = {
    { .chunks_per_epoch = 1,
      .covering_count = 1,
      .max_comp_chunk_bytes = 70,
      .cps_inner = 1,
      .num_shards = 1,
      .active_count_max = 1,
      .page_size = page,
      .chunks_per_shard_append = 2 },
  };
  uint32_t active_count[1] = { 1 };
  struct batch_aggregate_layout layout;
  CHECK(Fail,
        batch_aggregate_layout_init_compact(
          &layout, per_lod, active_count, 1) == 0);

  size_t capacity = 0;
  size_t run_capacity = 0;
  CHECK(Fail,
        host_batch_compact_capacity(per_lod,
                                    active_count,
                                    1,
                                    HOST_DELIVERY_FIXED_TAIL,
                                    page,
                                    &capacity,
                                    &run_capacity) == 0);
  CHECK(Fail, run_capacity == 1);
  host_data = (uint8_t*)platform_aligned_alloc(page, capacity);
  CHECK(Fail, host_data);

  test_sink_init(&sink, 2, 512);
  sink.shard_alignment = page;
  struct d2h_transfer_span span[1];
  size_t span_count = 0;
  size_t offsets[2] = { 0, 70 };
  size_t sizes[2] = { 70, 0 };
  uint8_t device[70];
  memset(device, 0xA1, sizeof(device));

  CHECK(Fail,
        host_batch_build_compact(&batch,
                                 host_data,
                                 capacity,
                                 offsets,
                                 sizes,
                                 &layout,
                                 per_lod,
                                 shards,
                                 active_count,
                                 1,
                                 HOST_DELIVERY_FIXED_TAIL,
                                 page,
                                 span,
                                 1,
                                 &span_count,
                                 host_data) == 0);
  CHECK(Fail, span_count == 1 && batch.run_count == 1);
  CHECK(Fail, !batch.runs[0].finalizes && batch.runs[0].tail_bytes == 0);
  copy_planned_spans(host_data, device, span, span_count);
  CHECK(Fail,
        deliver_host_batch(&batch, shards, &sink.base, page, NULL, NULL) == 0);
  CHECK(Fail, shard.epoch_in_shard == 1 && shard.shard_epoch == 0);
  CHECK(Fail, active.data_cursor == page && active.tail_bytes == 6);
  CHECK(Fail, memcmp(active.tail_buf, device + page, 6) == 0);
  CHECK(Fail, sink.write_direct_count == 1 && sink.finalize_count == 0);

  offsets[1] = 10;
  sizes[0] = 10;
  memset(device, 0xB2, 10);
  CHECK(Fail,
        host_batch_build_compact(&batch,
                                 host_data,
                                 capacity,
                                 offsets,
                                 sizes,
                                 &layout,
                                 per_lod,
                                 shards,
                                 active_count,
                                 1,
                                 HOST_DELIVERY_FIXED_TAIL,
                                 page,
                                 span,
                                 1,
                                 &span_count,
                                 host_data) == 0);
  CHECK(Fail, span_count == 1 && batch.runs[0].finalizes);
  CHECK(Fail, batch.runs[0].tail_bytes == 6);
  CHECK(Fail, memcmp(batch.runs[0].data, tail_buf, 6) == 0);
  copy_planned_spans(host_data, device, span, span_count);
  CHECK(Fail,
        deliver_host_batch(&batch, shards, &sink.base, page, NULL, NULL) == 0);

  struct test_shard_writer* writer = &sink.writers[0][0];
  CHECK(Fail, writer->finalized && sink.finalize_count == 1);
  CHECK(Fail, writer->size == 70 + 10 + sizeof(index) + 4);
  for (size_t i = 0; i < 70; ++i)
    CHECK(Fail, writer->buf[i] == 0xA1);
  for (size_t i = 70; i < 80; ++i)
    CHECK(Fail, writer->buf[i] == 0xB2);
  uint64_t stored_index[4] = { 0 };
  memcpy(stored_index, writer->buf + 80, sizeof(stored_index));
  CHECK(Fail, stored_index[0] == 0 && stored_index[1] == 70);
  CHECK(Fail, stored_index[2] == 70 && stored_index[3] == 10);
  CHECK(Fail, shard.epoch_in_shard == 0 && shard.shard_epoch == 1);
  CHECK(Fail, active.tail_bytes == 0 && active.writer == NULL);

  // The next generation exercises an exact-page non-final update, followed
  // by an empty final update. The exact page is written immediately and the
  // empty update contributes only the compact footer.
  offsets[1] = page;
  sizes[0] = page;
  memset(device, 0xC3, page);
  CHECK(Fail,
        host_batch_build_compact(&batch,
                                 host_data,
                                 capacity,
                                 offsets,
                                 sizes,
                                 &layout,
                                 per_lod,
                                 shards,
                                 active_count,
                                 1,
                                 HOST_DELIVERY_FIXED_TAIL,
                                 page,
                                 span,
                                 1,
                                 &span_count,
                                 host_data) == 0);
  copy_planned_spans(host_data, device, span, span_count);
  CHECK(Fail,
        deliver_host_batch(&batch, shards, &sink.base, page, NULL, NULL) == 0);
  CHECK(Fail, active.data_cursor == page && active.tail_bytes == 0);
  CHECK(Fail, shard.epoch_in_shard == 1 && shard.shard_epoch == 1);

  offsets[1] = 0;
  sizes[0] = 0;
  CHECK(Fail,
        host_batch_build_compact(&batch,
                                 host_data,
                                 capacity,
                                 offsets,
                                 sizes,
                                 &layout,
                                 per_lod,
                                 shards,
                                 active_count,
                                 1,
                                 HOST_DELIVERY_FIXED_TAIL,
                                 page,
                                 span,
                                 1,
                                 &span_count,
                                 host_data) == 0);
  CHECK(Fail, span_count == 0);
  CHECK(Fail,
        deliver_host_batch(&batch, shards, &sink.base, page, NULL, NULL) == 0);
  writer = &sink.writers[0][1];
  CHECK(Fail, writer->finalized && writer->size == page + sizeof(index) + 4);
  for (size_t i = 0; i < page; ++i)
    CHECK(Fail, writer->buf[i] == 0xC3);
  memcpy(stored_index, writer->buf + page, sizeof(stored_index));
  CHECK(Fail, stored_index[0] == 0 && stored_index[1] == page);
  CHECK(Fail, stored_index[2] == UINT64_MAX && stored_index[3] == UINT64_MAX);
  CHECK(Fail, shard.epoch_in_shard == 0 && shard.shard_epoch == 2);

  host_batch_destroy(&batch);
  test_sink_free(&sink);
  platform_aligned_free(host_data);
  platform_aligned_free(footer);
  return 0;

Fail:
  host_batch_destroy(&batch);
  test_sink_free(&sink);
  platform_aligned_free(host_data);
  platform_aligned_free(footer);
  return 1;
}

static int
test_indexed_padded_delivery(void)
{
  const size_t page = 64;
  uint8_t* footer = NULL;
  uint8_t* host_data = NULL;
  struct host_batch batch = { 0 };
  struct test_shard_sink sink;
  memset(&sink, 0, sizeof(sink));

  uint64_t index[10];
  memset(index, 0xFF, sizeof(index));
  const size_t footer_capacity = footer_capacity_for(5, page);
  footer = (uint8_t*)platform_aligned_alloc(page, footer_capacity);
  CHECK(Fail, footer);
  struct active_shard active = {
    .index = index,
    .footer_buf = footer,
  };
  struct shard_state shard = {
    .shard_inner_count = 1,
    .chunks_per_shard_inner = 1,
    .chunks_per_shard_total = 5,
    .chunks_per_shard_append = 5,
    .shards = &active,
    .footer_capacity = footer_capacity,
    .shard_file_capacity = 1024,
  };
  struct shard_state* shards[1] = { &shard };
  struct aggregate_layout per_lod[1] = {
    { .chunks_per_epoch = 1,
      .covering_count = 1,
      .max_comp_chunk_bytes = 130,
      .cps_inner = 1,
      .num_shards = 1,
      .active_count_max = 1,
      .page_size = page,
      .chunks_per_shard_append = 5 },
  };
  const uint32_t active_count[1] = { 1 };
  struct batch_aggregate_layout layout;
  CHECK(Fail,
        batch_aggregate_layout_init_compact(
          &layout, per_lod, active_count, 1) == 0);
  size_t capacity = 0;
  size_t run_capacity = 0;
  CHECK(Fail,
        host_batch_compact_capacity(per_lod,
                                    active_count,
                                    1,
                                    HOST_DELIVERY_INDEXED_PADDED,
                                    page,
                                    &capacity,
                                    &run_capacity) == 0);
  CHECK(Fail, run_capacity == 1);
  host_data = (uint8_t*)platform_aligned_alloc(page, capacity);
  CHECK(Fail, host_data);
  uint8_t device[130];
  struct d2h_transfer_span spans[1];
  struct stream_metrics metrics = { 0 };
  test_sink_init(&sink, 1, 1024);
  sink.shard_alignment = page;

  const size_t payloads[5] = { 0, 10, 64, 130, 10 };
  const uint8_t fills[5] = { 0, 0x11, 0x22, 0x33, 0x44 };
  for (size_t update = 0; update < 5; ++update) {
    const size_t payload = payloads[update];
    size_t offsets[2] = { 0, payload };
    size_t sizes[2] = { payload, 0 };
    memset(device, fills[update], payload);
    size_t span_count = 0;
    CHECK(Fail,
          host_batch_build_compact(&batch,
                                   host_data,
                                   capacity,
                                   offsets,
                                   sizes,
                                   &layout,
                                   per_lod,
                                   shards,
                                   active_count,
                                   1,
                                   HOST_DELIVERY_INDEXED_PADDED,
                                   page,
                                   spans,
                                   1,
                                   &span_count,
                                   host_data) == 0);
    CHECK(Fail, span_count == (payload > 0 ? 1u : 0u));
    copy_planned_spans(host_data, device, spans, span_count);
    CHECK(Fail,
          deliver_host_batch(
            &batch, shards, &sink.base, page, NULL, &metrics) == 0);
    if (update == 0) {
      CHECK(Fail, sink.open_count == 0 && shard.epoch_in_shard == 1);
      CHECK(Fail, metrics.shard_padding_physical_update_count == 0);
    }
  }

  struct test_shard_writer* writer = &sink.writers[0][0];
  CHECK(Fail, writer->finalized && writer->size == 414);
  CHECK(Fail, shard.epoch_in_shard == 0 && shard.shard_epoch == 1);
  CHECK(Fail, index[0] == UINT64_MAX && index[1] == UINT64_MAX);
  // The final ten payload bytes occupy [320, 330); the stored index follows.
  uint64_t stored[10];
  memcpy(stored, writer->buf + 330, sizeof(stored));
  CHECK(Fail, stored[0] == UINT64_MAX && stored[1] == UINT64_MAX);
  CHECK(Fail, stored[2] == 0 && stored[3] == 10);
  CHECK(Fail, stored[4] == 64 && stored[5] == 64);
  CHECK(Fail, stored[6] == 128 && stored[7] == 130);
  CHECK(Fail, stored[8] == 320 && stored[9] == 10);
  for (size_t i = 10; i < 64; ++i)
    CHECK(Fail, writer->buf[i] == 0);
  for (size_t i = 258; i < 320; ++i)
    CHECK(Fail, writer->buf[i] == 0);
  CHECK(Fail, metrics.shard_padding_logical_payload_bytes == 214);
  CHECK(Fail, metrics.shard_padding_internal_bytes == 116);
  CHECK(Fail, metrics.shard_padding_physical_update_count == 4);
  CHECK(Fail, metrics.shard_padding_padded_update_count == 2);
  CHECK(Fail,
        metrics.shard_padding_logical_payload_bytes +
            metrics.shard_padding_internal_bytes ==
          330);

  host_batch_destroy(&batch);
  test_sink_free(&sink);
  platform_aligned_free(host_data);
  platform_aligned_free(footer);
  return 0;

Fail:
  host_batch_destroy(&batch);
  test_sink_free(&sink);
  platform_aligned_free(host_data);
  platform_aligned_free(footer);
  return 1;
}

static int
test_indexed_compact_delivery(void)
{
  uint8_t* host_data = NULL;
  struct host_batch batch = { 0 };
  struct test_shard_sink sink;
  memset(&sink, 0, sizeof(sink));

  uint64_t index[4];
  memset(index, 0xFF, sizeof(index));
  struct active_shard active = { .index = index };
  struct shard_state shard = {
    .shard_inner_count = 1,
    .chunks_per_shard_inner = 1,
    .chunks_per_shard_total = 2,
    .chunks_per_shard_append = 2,
    .shards = &active,
    .shard_file_capacity = 128,
  };
  struct shard_state* shards[1] = { &shard };
  struct aggregate_layout per_lod[1] = {
    { .chunks_per_epoch = 1,
      .covering_count = 1,
      .max_comp_chunk_bytes = 10,
      .cps_inner = 1,
      .num_shards = 1,
      .active_count_max = 1,
      .page_size = 0,
      .chunks_per_shard_append = 2 },
  };
  const uint32_t active_count[1] = { 1 };
  struct batch_aggregate_layout layout;
  CHECK(Fail,
        batch_aggregate_layout_init_compact(
          &layout, per_lod, active_count, 1) == 0);
  size_t capacity = 0;
  size_t run_capacity = 0;
  CHECK(Fail,
        host_batch_compact_capacity(per_lod,
                                    active_count,
                                    1,
                                    HOST_DELIVERY_INDEXED_COMPACT,
                                    0,
                                    &capacity,
                                    &run_capacity) == 0);
  host_data = (uint8_t*)malloc(capacity);
  CHECK(Fail, host_data);
  struct d2h_transfer_span span[1];
  struct stream_metrics metrics = { 0 };
  test_sink_init(&sink, 1, 128);

  for (size_t update = 0; update < 2; ++update) {
    const size_t payload = update == 0 ? 0 : 10;
    size_t offsets[2] = { 0, payload };
    size_t sizes[2] = { payload, 0 };
    size_t span_count = 0;
    CHECK(Fail,
          host_batch_build_compact(&batch,
                                   host_data,
                                   capacity,
                                   offsets,
                                   sizes,
                                   &layout,
                                   per_lod,
                                   shards,
                                   active_count,
                                   1,
                                   HOST_DELIVERY_INDEXED_COMPACT,
                                   0,
                                   span,
                                   1,
                                   &span_count,
                                   host_data) == 0);
    if (payload > 0) {
      memset(host_data, 0x5A, payload);
      CHECK(Fail, span_count == 1 && span[0].bytes == payload);
    } else {
      CHECK(Fail, span_count == 0);
    }
    CHECK(Fail,
          deliver_host_batch(&batch, shards, &sink.base, 0, NULL, &metrics) ==
            0);
  }

  struct test_shard_writer* writer = &sink.writers[0][0];
  CHECK(Fail, writer->finalized && writer->size == 46);
  for (size_t i = 0; i < 10; ++i)
    CHECK(Fail, writer->buf[i] == 0x5A);
  uint64_t stored[4];
  memcpy(stored, writer->buf + 10, sizeof(stored));
  CHECK(Fail, stored[0] == UINT64_MAX && stored[1] == UINT64_MAX);
  CHECK(Fail, stored[2] == 0 && stored[3] == 10);
  CHECK(Fail, sink.write_direct_count == 1 && sink.write_count == 1);
  CHECK(Fail, metrics.shard_padding_logical_payload_bytes == 10);
  CHECK(Fail, metrics.shard_padding_internal_bytes == 0);
  CHECK(Fail, metrics.shard_padding_physical_update_count == 1);
  CHECK(Fail, metrics.shard_padding_padded_update_count == 0);

  host_batch_destroy(&batch);
  test_sink_free(&sink);
  free(host_data);
  return 0;

Fail:
  host_batch_destroy(&batch);
  test_sink_free(&sink);
  free(host_data);
  return 1;
}

static int
test_shard_drainer_commands(void)
{
  const size_t page = 64;
  uint8_t* data = NULL;
  uint8_t* footer = NULL;
  struct shard_drainer drain = { 0 };

  data = (uint8_t*)platform_aligned_alloc(page, 128);
  footer = (uint8_t*)platform_aligned_alloc(page, 64);
  CHECK(Fail, data && footer);
  memset(data, 0xAB, 70);
  memset(data + 70, 0, 58);
  uint64_t index[4];
  memset(index, 0xFF, sizeof(index));
  struct active_shard active = {
    .index = index,
    .footer_buf = footer,
  };
  struct shard_state shard = {
    .shard_inner_count = 1,
    .chunks_per_shard_inner = 1,
    .chunks_per_shard_total = 1,
    .chunks_per_shard_append = 1,
    .shards = &active,
    .footer_capacity = 64,
  };
  struct shard_state* shards[1] = { &shard };
  size_t offsets[1] = { 0 };
  size_t sizes[1] = { 70 };
  struct host_batch_run run = {
    .level = 0,
    .inner_shard = 0,
    .flat_shard = 0,
    .active_count = 1,
    .epoch_in_shard = 0,
    .chunks_per_shard_inner = 1,
    .finalizes = 1,
    .ends_generation_run = 1,
    .data = data,
    .page_size = page,
    .payload_bytes = 70,
    .source_offset = 0,
    .offsets = offsets,
    .chunk_sizes = sizes,
  };
  struct host_batch host = {
    .runs = &run,
    .run_count = 1,
    .run_capacity = 1,
    .nlod = 1,
    .policy = HOST_DELIVERY_INDEXED_PADDED,
    .shard_alignment = page,
    .slot_lifetime = data,
  };
  struct shard_drain_command command;
  CHECK(Fail, shard_drain_begin(&drain, &host, shards) == 0);

  CHECK(Fail, shard_drain_next(&drain, &command) == 1);
  CHECK(Fail, command.kind == SHARD_DRAIN_DATA);
  CHECK(Fail, command.file_offset == 0 && command.physical_bytes == 64);
  CHECK(Fail, command.direct_write_eligible);
  CHECK(Fail, command.buffer_lease.kind == SHARD_DRAIN_LEASE_HOST_BATCH);
  CHECK(Fail, command.logical_payload_bytes == 70);
  CHECK(Fail, shard_drain_accept(&drain, &command) == 0);
  CHECK(Fail, active.data_cursor == 64);

  CHECK(Fail, shard_drain_next(&drain, &command) == 1);
  CHECK(Fail, command.kind == SHARD_DRAIN_FOOTER);
  CHECK(Fail, command.file_offset == 64 && command.physical_bytes == 64);
  CHECK(Fail, command.buffer_lease.kind == SHARD_DRAIN_LEASE_FOOTER);
  CHECK(Fail,
        command.source_begin == footer && command.source_end == footer + 64);
  CHECK(Fail, command.logical_payload_bytes == 0);
  CHECK(Fail, shard_drain_prepare(&drain, &command) == 0);
  CHECK(Fail,
        command.source_begin == footer && command.source_end == footer + 64);
  for (size_t i = 0; i < 6; ++i)
    CHECK(Fail, footer[i] == 0xAB);
  uint64_t stored[2];
  memcpy(stored, footer + 6, sizeof(stored));
  CHECK(Fail, stored[0] == 0 && stored[1] == 70);
  CHECK(Fail, shard_drain_accept(&drain, &command) == 0);
  CHECK(Fail, index[0] == 0 && index[1] == 70);

  CHECK(Fail, shard_drain_next(&drain, &command) == 1);
  CHECK(Fail,
        command.kind == SHARD_DRAIN_TRUNCATE && command.logical_size == 90);
  CHECK(Fail, shard_drain_accept(&drain, &command) == 0);
  CHECK(Fail, shard_drain_next(&drain, &command) == 1);
  CHECK(Fail,
        command.kind == SHARD_DRAIN_FINALIZE && command.closes_generation);
  CHECK(Fail, shard_drain_accept(&drain, &command) == 0);
  CHECK(Fail, shard_drain_next(&drain, &command) == 0);
  CHECK(Fail, shard.shard_epoch == 1 && shard.epoch_in_shard == 0);
  CHECK(Fail, active.data_cursor == 0 && active.writer == NULL);
  shard_drain_destroy(&drain);

  // A rejected sink command is sticky and leaves that command's state
  // uncommitted. Previously accepted commands would deliberately stay put.
  memset(index, 0xFF, sizeof(index));
  shard = (struct shard_state){
    .shard_inner_count = 1,
    .chunks_per_shard_inner = 1,
    .chunks_per_shard_total = 2,
    .chunks_per_shard_append = 2,
    .shards = &active,
    .footer_capacity = 64,
  };
  active.data_cursor = 0;
  run.finalizes = 0;
  run.ends_generation_run = 1;
  run.payload_bytes = 10;
  sizes[0] = 10;
  CHECK(Fail, shard_drain_begin(&drain, &host, shards) == 0);
  CHECK(Fail, shard_drain_next(&drain, &command) == 1);
  CHECK(Fail,
        command.kind == SHARD_DRAIN_DATA && command.physical_bytes == page);
  shard_drain_abort(&drain);
  CHECK(Fail, shard_drain_next(&drain, &command) == -1);
  CHECK(Fail, shard_drain_accept(&drain, &command) != 0);
  CHECK(Fail, active.data_cursor == 0 && shard.epoch_in_shard == 0);
  CHECK(Fail, index[0] == UINT64_MAX && index[1] == UINT64_MAX);
  shard_drain_destroy(&drain);

  // Empty indexed updates yield a zero-byte command so state remains
  // acceptance-gated, but the executor can skip opening or writing a file.
  memset(index, 0xFF, sizeof(index));
  active.data_cursor = 0;
  shard.epoch_in_shard = 0;
  run.payload_bytes = 0;
  sizes[0] = 0;
  CHECK(Fail, shard_drain_begin(&drain, &host, shards) == 0);
  CHECK(Fail, shard_drain_next(&drain, &command) == 1);
  CHECK(Fail,
        command.kind == SHARD_DRAIN_DATA && command.physical_bytes == 0 &&
          !command.counts_physical_update);
  CHECK(Fail, shard.epoch_in_shard == 0 && active.data_cursor == 0);
  CHECK(Fail, shard_drain_accept(&drain, &command) == 0);
  CHECK(Fail, shard.epoch_in_shard == 1 && active.data_cursor == 0);
  CHECK(Fail, shard_drain_next(&drain, &command) == 0);
  shard_drain_destroy(&drain);

  // Physical cursor arithmetic is checked before a command is exposed.
  active.data_cursor = UINT64_MAX - 32;
  shard.epoch_in_shard = 0;
  run.payload_bytes = 10;
  sizes[0] = 10;
  CHECK(Fail, shard_drain_begin(&drain, &host, shards) == 0);
  CHECK(Fail, shard_drain_next(&drain, &command) == -1);
  CHECK(Fail, active.data_cursor == UINT64_MAX - 32);
  CHECK(Fail, shard.epoch_in_shard == 0);
  CHECK(Fail, shard_drain_next(&drain, &command) == -1);
  shard_drain_destroy(&drain);

  platform_aligned_free(data);
  platform_aligned_free(footer);
  return 0;

Fail:
  shard_drain_destroy(&drain);
  platform_aligned_free(data);
  platform_aligned_free(footer);
  return 1;
}

int
main(void)
{
  return test_span_planning() || test_host_batch_runs() ||
         test_compact_layout_fixed_index() || test_compact_run_planning() ||
         test_compact_extent_edges() || test_compact_host_tail_delivery() ||
         test_indexed_padded_delivery() || test_indexed_compact_delivery() ||
         test_shard_drainer_commands();
}
