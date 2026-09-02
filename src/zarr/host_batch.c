#include "zarr/host_batch.h"

#include "defs.limits.h"
#include "stream/host_output_pool.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

#include <stdlib.h>
#include <string.h>

enum host_batch_storage
host_batch_storage_select(int fixed_size, size_t shard_alignment)
{
  if (fixed_size)
    return HOST_BATCH_FIXED_SIZE;
  return shard_alignment > 0 ? HOST_BATCH_PAGE_PADDED : HOST_BATCH_PACKED;
}

static int
host_batch_reserve(struct host_batch* host, size_t count)
{
  if (count <= host->run_capacity)
    return 0;
  CHECK_MUL_OVERFLOW(Error, count, sizeof(*host->runs), SIZE_MAX);
  struct host_batch_run* p =
    (struct host_batch_run*)realloc(host->runs, count * sizeof(*host->runs));
  CHECK(Error, p);
  host->runs = p;
  host->run_capacity = count;
  return 0;

Error:
  return 1;
}

static int
run_payload_bytes(const size_t* chunk_sizes, uint64_t count, size_t* out_bytes)
{
  size_t bytes = 0;
  for (uint64_t i = 0; i < count; ++i) {
    CHECK(Error, chunk_sizes[i] <= SIZE_MAX - bytes);
    bytes += chunk_sizes[i];
  }
  *out_bytes = bytes;
  return 0;

Error:
  return 1;
}

static int
checked_align_up_size(size_t value, size_t alignment, size_t* out)
{
  CHECK(Error, out && alignment > 0);
  const size_t rem = value % alignment;
  const size_t add = rem == 0 ? 0 : alignment - rem;
  CHECK(Error, value <= SIZE_MAX - add);
  *out = value + add;
  return 0;

Error:
  return 1;
}

static int
maximum_generation_runs(uint32_t active,
                        uint64_t chunks_per_shard_append,
                        uint64_t* out)
{
  CHECK(Error, out && chunks_per_shard_append > 0);
  if (active == 0) {
    *out = 0;
    return 0;
  }
  // The worst committed starting point has one append epoch left in the
  // current generation, followed by full generations.
  const uint64_t left = (uint64_t)active - 1;
  *out =
    1 + left / chunks_per_shard_append + (left % chunks_per_shard_append != 0);
  return 0;

Error:
  return 1;
}

int
host_batch_capacity(const struct aggregate_layout* level_layouts,
                    const uint32_t* active_count_by_level,
                    uint8_t nlod,
                    enum host_batch_storage storage,
                    size_t shard_alignment,
                    size_t* out_bytes,
                    size_t* out_run_count)
{
  CHECK(Error, level_layouts && active_count_by_level);
  CHECK(Error, out_bytes && out_run_count);
  CHECK(Error, nlod >= 1 && nlod <= LOD_MAX_LEVELS);

  size_t bytes = 0;
  size_t runs = 0;
  for (uint8_t lv = 0; lv < nlod; ++lv) {
    const struct aggregate_layout* al = &level_layouts[lv];
    const uint32_t active = active_count_by_level[lv];
    CHECK(Error, al->page_size == shard_alignment);
    CHECK(Error, al->cps_inner > 0 && al->num_shards > 0);
    CHECK_MUL_OVERFLOW(Error, al->num_shards, al->cps_inner, UINT64_MAX);
    CHECK(Error, al->covering_count == al->num_shards * al->cps_inner);
    CHECK(Error, al->chunks_per_shard_append > 0);
    CHECK_MUL_OVERFLOW(Error, active, al->chunks_per_epoch, UINT64_MAX);
    const uint64_t chunks = (uint64_t)active * al->chunks_per_epoch;
    CHECK_MUL_OVERFLOW(Error, chunks, al->max_comp_chunk_bytes, SIZE_MAX);
    const size_t payload = (size_t)chunks * al->max_comp_chunk_bytes;

    uint64_t generation_runs = 0;
    CHECK(Error,
          maximum_generation_runs(
            active, al->chunks_per_shard_append, &generation_runs) == 0);
    CHECK_MUL_OVERFLOW(Error, generation_runs, al->num_shards, SIZE_MAX);
    const size_t physical_runs = (size_t)(generation_runs * al->num_shards);
    CHECK(Error, physical_runs <= SIZE_MAX - runs);
    runs += physical_runs;

    if (shard_alignment == 0) {
      CHECK(Error,
            storage == HOST_BATCH_FIXED_SIZE || storage == HOST_BATCH_PACKED);
      CHECK(Error, payload <= SIZE_MAX - bytes);
      bytes += payload;
      continue;
    }

    CHECK(Error, shard_alignment > 0);
    CHECK(Error,
          storage == HOST_BATCH_FIXED_SIZE ||
            storage == HOST_BATCH_PAGE_PADDED);

    // Each physical run may start alignment-1 bytes after the previous one.
    // Fixed runs may also prepend at most alignment-1 committed tail bytes;
    // variable-size runs may retain at most alignment-1 zero bytes after
    // payload. The same checked bound covers both storage choices without
    // knowing the committed starting epoch or the per-run payload distribution.
    const size_t per_run_slack = shard_alignment - 1;
    CHECK_MUL_OVERFLOW(Error, physical_runs, per_run_slack, SIZE_MAX);
    const size_t one_slack_each = physical_runs * per_run_slack;
    CHECK(Error, one_slack_each <= SIZE_MAX / 2);
    const size_t slack = one_slack_each * 2;
    CHECK(Error, payload <= SIZE_MAX - slack);
    const size_t level_bytes = payload + slack;
    CHECK(Error, level_bytes <= SIZE_MAX - bytes);
    bytes += level_bytes;
  }

  *out_bytes = bytes;
  *out_run_count = runs;
  return 0;

Error:
  if (out_bytes)
    *out_bytes = 0;
  if (out_run_count)
    *out_run_count = 0;
  return 1;
}

int
host_batch_build(struct host_batch* host,
                 void* aggregate_data,
                 size_t aggregate_capacity,
                 const size_t* offsets,
                 const size_t* chunk_sizes,
                 const struct batch_aggregate_layout* batch_layout,
                 const struct aggregate_layout* level_layouts,
                 struct shard_state* const* shards_by_level,
                 const uint32_t* active_count_by_level,
                 enum host_batch_storage storage,
                 size_t shard_alignment,
                 struct d2h_transfer_span* spans,
                 size_t span_capacity,
                 size_t* out_span_count)
{
  CHECK(Error, host && aggregate_data && offsets && chunk_sizes);
  CHECK(Error, batch_layout && level_layouts && shards_by_level);
  CHECK(Error, active_count_by_level && out_span_count);
  CHECK(Error, batch_layout->nlod >= 1 && batch_layout->nlod <= LOD_MAX_LEVELS);
  const uint8_t nlod = batch_layout->nlod;
  CHECK(Error,
        shard_alignment == 0
          ? storage == HOST_BATCH_FIXED_SIZE || storage == HOST_BATCH_PACKED
          : storage == HOST_BATCH_FIXED_SIZE ||
              storage == HOST_BATCH_PAGE_PADDED);
  CHECK(Error, batch_layout->total_batch_covering <= SIZE_MAX - nlod);
  const size_t metadata_entries =
    (size_t)batch_layout->total_batch_covering + nlod;

  size_t run_count = 0;
  for (uint8_t lv = 0; lv < nlod; ++lv) {
    const struct shard_state* ss = shards_by_level[lv];
    CHECK(Error, ss && ss->chunks_per_shard_append > 0);
    CHECK(Error, ss->epoch_in_shard < ss->chunks_per_shard_append);
    uint32_t left = active_count_by_level[lv];
    uint64_t epoch = ss->epoch_in_shard;
    while (left > 0) {
      const uint64_t remaining = ss->chunks_per_shard_append - epoch;
      const uint32_t run = left < remaining ? left : (uint32_t)remaining;
      CHECK(Error, run > 0);
      CHECK(Error, ss->shard_inner_count <= SIZE_MAX - run_count);
      run_count += (size_t)ss->shard_inner_count;
      left -= run;
      epoch = run == remaining ? 0 : epoch + run;
    }
  }
  CHECK(Error, host_batch_reserve(host, run_count) == 0);
  CHECK(Error, !spans || span_capacity >= run_count);

  host->run_count = 0;
  host->nlod = nlod;
  host->storage = storage;
  host->shard_alignment = shard_alignment;
  *out_span_count = 0;
  size_t host_cursor = 0;
  const uintptr_t host_base = (uintptr_t)aggregate_data;

  for (uint8_t lv = 0; lv < nlod; ++lv) {
    struct shard_state* ss = shards_by_level[lv];
    const struct aggregate_layout* al = &level_layouts[lv];
    const struct lod_segment* seg = &batch_layout->lods[lv];
    const uint32_t n_active = active_count_by_level[lv];
    const uint64_t cps = ss->chunks_per_shard_inner;
    CHECK(Error,
          seg->batch_covering_offset <= batch_layout->total_batch_covering);
    CHECK(Error, seg->batch_covering_offset <= SIZE_MAX - lv);
    const size_t metadata_base = seg->batch_covering_offset + (size_t)lv;
    const size_t page_size = shard_alignment;
    CHECK(Error, al->page_size == shard_alignment);
    CHECK(Error, al->page_size == level_layouts[0].page_size);
    CHECK(Error, cps == seg->chunks_per_shard_inner);
    CHECK(Error, cps == al->cps_inner);
    CHECK(Error, ss->shard_inner_count == al->num_shards);
    CHECK(Error, ss->chunks_per_shard_append == al->chunks_per_shard_append);
    CHECK(Error, seg->n_active == n_active);
    CHECK(Error, seg->covering_count == al->covering_count);
    CHECK(Error, seg->chunks_per_epoch == al->chunks_per_epoch);
    CHECK_MUL_OVERFLOW(Error, n_active, al->covering_count, SIZE_MAX);
    const size_t segment_entries = (size_t)n_active * al->covering_count;
    CHECK(Error, metadata_base < metadata_entries);
    CHECK(Error, segment_entries <= metadata_entries - metadata_base - 1);
    CHECK(Error, seg->data_segment_offset <= batch_layout->total_data_bytes);
    CHECK(Error,
          seg->data_segment_bytes <=
            batch_layout->total_data_bytes - seg->data_segment_offset);

    uint32_t a = 0;
    uint64_t epoch = ss->epoch_in_shard;
    uint64_t generation = ss->shard_epoch;
    while (a < n_active) {
      const uint64_t remaining = ss->chunks_per_shard_append - epoch;
      const uint32_t left = n_active - a;
      const uint32_t run_len = left < remaining ? left : (uint32_t)remaining;
      const int finalizes = run_len == remaining;

      for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
        const uint64_t j = si * (uint64_t)n_active * cps + (uint64_t)a * cps;
        const uint64_t nchunks = (uint64_t)run_len * cps;
        CHECK(Error, j <= segment_entries);
        CHECK(Error, nchunks <= segment_entries - (size_t)j);
        size_t payload = 0;
        CHECK(Error,
              run_payload_bytes(
                chunk_sizes + metadata_base + j, nchunks, &payload) == 0);
        const size_t source = offsets[metadata_base + j];
        // Variable-size compact aggregates pack actual bytes across LOD
        // boundaries; a later LOD can therefore begin before its worst-case
        // capacity offset in the static layout.
        CHECK(Error, source <= batch_layout->total_data_bytes);
        CHECK(Error, payload <= batch_layout->total_data_bytes - source);

        size_t region = host_cursor;
        size_t reserve = payload;
        const size_t tail = storage == HOST_BATCH_FIXED_SIZE && a == 0
                              ? ss->shards[si].tail_bytes
                              : 0;
        CHECK(Error, tail <= SIZE_MAX - payload);
        const size_t logical_run = tail + payload;
        const int needs_region = logical_run > 0;
        if (page_size > 0 && needs_region) {
          CHECK(Error, storage != HOST_BATCH_PACKED);
          CHECK(Error, tail < page_size);
          const size_t address_remainder = (host_base + region) % page_size;
          if (address_remainder != 0) {
            const size_t pad = page_size - address_remainder;
            CHECK(Error, region <= SIZE_MAX - pad);
            region += pad;
          }
          if (storage == HOST_BATCH_FIXED_SIZE) {
            CHECK(Error, logical_run >= payload);
            reserve = logical_run;
          } else {
            CHECK(Error, storage == HOST_BATCH_PAGE_PADDED);
            CHECK(Error,
                  checked_align_up_size(payload, page_size, &reserve) == 0);
          }
        } else if (!needs_region) {
          reserve = 0;
        } else {
          CHECK(Error,
                storage == HOST_BATCH_FIXED_SIZE ||
                  storage == HOST_BATCH_PACKED);
          CHECK(Error, tail == 0);
        }
        CHECK(Error, region <= aggregate_capacity);
        CHECK(Error, reserve <= aggregate_capacity - region);
        CHECK(Error, tail <= reserve);
        CHECK(Error, payload <= reserve - tail);
        if (tail > 0) {
          CHECK(Error, ss->shards[si].tail_buf);
          memcpy(
            (uint8_t*)aggregate_data + region, ss->shards[si].tail_buf, tail);
        }
        if (storage == HOST_BATCH_PAGE_PADDED && reserve > payload)
          memset(
            (uint8_t*)aggregate_data + region + payload, 0, reserve - payload);

        if (payload > 0 && spans) {
          spans[*out_span_count] = (struct d2h_transfer_span){
            .device_offset = source,
            .host_offset = region + tail,
            .bytes = payload,
          };
          (*out_span_count)++;
        }

        CHECK(Error, generation <= (UINT64_MAX - si) / ss->shard_inner_count);

        host->runs[host->run_count++] = (struct host_batch_run){
          .level = lv,
          .inner_shard = si,
          .flat_shard = generation * ss->shard_inner_count + si,
          .active_begin = a,
          .active_count = run_len,
          .epoch_in_shard = epoch,
          .chunks_per_shard_inner = cps,
          .finalizes = finalizes,
          .ends_generation_run = si + 1 == ss->shard_inner_count,
          .data = (uint8_t*)aggregate_data + region,
          .page_size = page_size,
          .tail_bytes = tail,
          .payload_bytes = payload,
          .source_offset = source,
          .offsets = offsets + metadata_base + j,
          .chunk_sizes = chunk_sizes + metadata_base + j,
        };
        CHECK(Error, region <= SIZE_MAX - reserve);
        host_cursor = region + reserve;
      }

      a += run_len;
      if (finalizes) {
        epoch = 0;
        CHECK(Error, generation < UINT64_MAX);
        generation++;
      } else {
        epoch += run_len;
      }
    }
  }

  CHECK(Error, host->run_count == run_count);
  return 0;

Error:
  if (host) {
    host->run_count = 0;
    host->nlod = 0;
    host->storage = HOST_BATCH_FIXED_SIZE;
    host->shard_alignment = 0;
  }
  if (out_span_count)
    *out_span_count = 0;
  return 1;
}

void
host_batch_destroy(struct host_batch* host)
{
  if (!host)
    return;
  if (host->output_group)
    host_output_group_seal(host->output_group);
  free(host->runs);
  *host = (struct host_batch){ 0 };
}
