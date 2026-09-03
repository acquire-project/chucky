#include "cpu/aggregate.h"

#include "threadpool/threadpool.h"
#include "util/prelude.h"
#include "zarr/host_batch.h"

#include <string.h>

int
aggregate_cpu_batch_prepare_unified(const struct aggregate_cpu_inputs* in)
{
  CHECK(Error, in && in->layout && in->ws);
  CHECK(Error, in->compressed_base && in->comp_sizes_base && in->gather);
  CHECK(Error, in->ws->perm && in->ws->permuted_sizes && in->ws->offsets);
  CHECK(Error, in->ws->chunk_sizes);
  CHECK(Error, in->layout->total_batch_covering <= SIZE_MAX - in->layout->nlod);
  const size_t metadata_count =
    (size_t)in->layout->total_batch_covering + in->layout->nlod;
  CHECK_MUL_OVERFLOW(
    Error, metadata_count, sizeof(*in->ws->permuted_sizes), SIZE_MAX);

  memset(in->ws->permuted_sizes,
         0,
         metadata_count * sizeof(*in->ws->permuted_sizes));
  for (uint64_t i = 0; i < in->layout->total_batch_chunks; ++i) {
    const uint32_t target = in->ws->perm[i];
    CHECK(Error, target < metadata_count);
    in->ws->permuted_sizes[target] = in->comp_sizes_base[in->gather[i]];
  }
  memcpy(in->ws->chunk_sizes,
         in->ws->permuted_sizes,
         metadata_count * sizeof(*in->ws->chunk_sizes));

  for (uint8_t lv = 0; lv < in->layout->nlod; ++lv) {
    const struct lod_segment* segment = &in->layout->lods[lv];
    const uint64_t base = segment->batch_covering_offset + lv;
    CHECK_MUL_OVERFLOW(
      Error, segment->n_active, segment->covering_count, UINT64_MAX);
    const uint64_t count =
      (uint64_t)segment->n_active * segment->covering_count;
    CHECK(Error, base < metadata_count);
    CHECK(Error, count <= metadata_count - base - 1);
    in->ws->offsets[base] = 0;
    for (uint64_t i = 0; i < count; ++i) {
      CHECK(Error,
            in->ws->permuted_sizes[base + i] <=
              SIZE_MAX - in->ws->offsets[base + i]);
      in->ws->offsets[base + i + 1] =
        in->ws->offsets[base + i] + in->ws->permuted_sizes[base + i];
    }
  }
  return 0;

Error:
  return 1;
}

struct gather_host_ctx
{
  const char* compressed;
  const size_t* comp_sizes;
  const uint32_t* gather;
  const uint32_t* perm;
  const size_t* destinations;
  char* data;
  size_t max_comp;
};

static void
gather_host_range(size_t beg, size_t end, int tid, void* vctx)
{
  (void)tid;
  struct gather_host_ctx* c = (struct gather_host_ctx*)vctx;
  for (size_t i = beg; i < end; ++i) {
    const uint32_t source = c->gather[i];
    const size_t nbytes = c->comp_sizes[source];
    if (nbytes == 0)
      continue;
    memcpy(c->data + c->destinations[c->perm[i]],
           c->compressed + (uint64_t)source * c->max_comp,
           nbytes);
  }
}

int
aggregate_cpu_batch_copy_to_host(const struct aggregate_cpu_inputs* in,
                                 const struct host_batch* host)
{
  CHECK(Error, in && in->layout && in->ws && host && in->pool);
  CHECK(Error, in->ws->data && in->ws->perm && in->ws->permuted_sizes);
  CHECK(Error, in->ws->offsets && in->ws->chunk_sizes);
  CHECK(Error, in->compressed_base && in->comp_sizes_base && in->gather);
  CHECK(Error, in->layout->total_batch_covering <= SIZE_MAX - in->layout->nlod);
  const size_t metadata_count =
    (size_t)in->layout->total_batch_covering + in->layout->nlod;
  memset(in->ws->permuted_sizes, 0xFF, metadata_count * sizeof(size_t));

  const uintptr_t output = (uintptr_t)in->ws->data;
  for (size_t r = 0; r < host->run_count; ++r) {
    const struct host_batch_run* run = &host->runs[r];
    CHECK(Error, run->offsets >= in->ws->offsets);
    const size_t metadata_base = (size_t)(run->offsets - in->ws->offsets);
    CHECK_MUL_OVERFLOW(
      Error, run->active_count, run->chunks_per_shard_inner, SIZE_MAX);
    const size_t chunks =
      (size_t)run->active_count * run->chunks_per_shard_inner;
    CHECK(Error, metadata_base <= metadata_count);
    CHECK(Error, chunks <= metadata_count - metadata_base);
    CHECK(Error, (uintptr_t)run->data >= output);
    const size_t run_offset = (size_t)((uintptr_t)run->data - output);
    CHECK(Error, run_offset <= in->ws->data_capacity);
    CHECK(Error, run->tail_bytes <= in->ws->data_capacity - run_offset);
    const size_t payload_offset = run_offset + run->tail_bytes;

    for (size_t j = 0; j < chunks; ++j) {
      CHECK(Error, run->offsets[j] >= run->source_offset);
      const size_t relative = run->offsets[j] - run->source_offset;
      CHECK(Error, relative <= run->payload_bytes);
      CHECK(Error, run->chunk_sizes[j] <= run->payload_bytes - relative);
      CHECK(Error, payload_offset <= in->ws->data_capacity);
      CHECK(Error, relative <= in->ws->data_capacity - payload_offset);
      const size_t destination = payload_offset + relative;
      CHECK(Error, run->chunk_sizes[j] <= in->ws->data_capacity - destination);
      in->ws->permuted_sizes[metadata_base + j] = destination;
    }
  }

  for (uint64_t i = 0; i < in->layout->total_batch_chunks; ++i) {
    const uint32_t target = in->ws->perm[i];
    CHECK(Error, target < metadata_count);
    CHECK(Error, in->ws->permuted_sizes[target] != SIZE_MAX);
    const size_t nbytes = in->comp_sizes_base[in->gather[i]];
    CHECK(Error, nbytes <= in->layout->max_comp_chunk_bytes);
    CHECK(Error,
          nbytes <= in->ws->data_capacity - in->ws->permuted_sizes[target]);
  }

  struct gather_host_ctx context = {
    .compressed = (const char*)in->compressed_base,
    .comp_sizes = in->comp_sizes_base,
    .gather = in->gather,
    .perm = in->ws->perm,
    .destinations = in->ws->permuted_sizes,
    .data = (char*)in->ws->data,
    .max_comp = in->layout->max_comp_chunk_bytes,
  };
  threadpool_for_n(
    in->pool, in->layout->total_batch_chunks, gather_host_range, &context);
  return 0;

Error:
  return 1;
}
