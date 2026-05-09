#include "aggregate_contract.h"

#include "log/log.h"
#include "stream/types.aggregate.h"

#include <inttypes.h>

int
verify_aggregate_result_carryover(const struct aggregate_result* result,
                                  const struct aggregate_layout* layout,
                                  const size_t* tail_in,
                                  uint32_t n_active)
{
  int violations = 0;
  if (!result || !layout) {
    log_error("contract: NULL result/layout");
    return 1;
  }
  if (layout->page_size == 0 || layout->shard_capacity == 0) {
    log_error("contract: carry-over verifier called with page_size=%zu "
              "shard_capacity=%zu (expected both > 0)",
              layout->page_size,
              layout->shard_capacity);
    return 1;
  }
  if (n_active == 0)
    return 0; // empty batch is trivially correct

  const uint64_t cps_inner = layout->cps_inner;
  const uint64_t num_shards = layout->num_shards;
  const size_t shard_capacity = layout->shard_capacity;
  const uint64_t span = (uint64_t)n_active * cps_inner;

  for (uint64_t si = 0; si < num_shards; ++si) {
    const size_t expected_tail = tail_in ? tail_in[si] : 0;
    const size_t shard_base = (size_t)si * shard_capacity;
    const size_t shard_end = shard_base + shard_capacity;
    const uint64_t j0 = si * span;

    // (3) First chunk of the shard anchored at shard_base + tail_in.
    const size_t first = result->offsets[j0];
    const size_t expected_first = shard_base + expected_tail;
    if (first != expected_first) {
      log_error("contract: shard %" PRIu64 " first offset=%zu, expected %zu "
                "(shard_base=%zu + tail_in=%zu)",
                si,
                first,
                expected_first,
                shard_base,
                expected_tail);
      ++violations;
    }

    // (4) Within-shard tight packing: offsets[j+1] - offsets[j] == size[j].
    for (uint64_t k = 0; k + 1 < span; ++k) {
      const uint64_t j = j0 + k;
      const size_t delta = result->offsets[j + 1] - result->offsets[j];
      const size_t sz = result->chunk_sizes[j];
      if (delta != sz) {
        log_error("contract: shard %" PRIu64 " chunk-pack: offsets[%" PRIu64
                  "+1]-offsets[%" PRIu64 "]=%zu, chunk_sizes[%" PRIu64 "]=%zu",
                  si,
                  j,
                  j,
                  delta,
                  j,
                  sz);
        ++violations;
      }
    }

    // (5) Each chunk fits inside the shard's reserved region.
    for (uint64_t k = 0; k < span; ++k) {
      const uint64_t j = j0 + k;
      const size_t chunk_end = result->offsets[j] + result->chunk_sizes[j];
      if (chunk_end > shard_end) {
        log_error("contract: shard %" PRIu64 " chunk %" PRIu64 " end=%zu "
                  "exceeds shard_end=%zu (capacity=%zu)",
                  si,
                  j,
                  chunk_end,
                  shard_end,
                  shard_capacity);
        ++violations;
      }
      if (result->offsets[j] < shard_base) {
        log_error("contract: shard %" PRIu64 " chunk %" PRIu64
                  " offset=%zu below shard_base=%zu",
                  si,
                  j,
                  result->offsets[j],
                  shard_base);
        ++violations;
      }
    }
  }

  return violations;
}

int
verify_aggregate_result_contiguous(const struct aggregate_result* result,
                                   const struct aggregate_layout* layout,
                                   uint32_t n_active)
{
  int violations = 0;
  if (!result || !layout) {
    log_error("contract: NULL result/layout");
    return 1;
  }
  if (layout->page_size != 0) {
    log_error("contract: contiguous verifier called with page_size=%zu "
              "(expected 0)",
              layout->page_size);
    return 1;
  }
  if (n_active == 0)
    return 0;

  const uint64_t cps_inner = layout->cps_inner;
  const uint64_t covering = (uint64_t)n_active * layout->covering_count;
  (void)cps_inner;

  // (2) Tightly-packed prefix sum starting at 0.
  if (result->offsets[0] != 0) {
    log_error("contract: contiguous offsets[0]=%zu, expected 0",
              result->offsets[0]);
    ++violations;
  }
  for (uint64_t j = 0; j + 1 < covering; ++j) {
    const size_t delta = result->offsets[j + 1] - result->offsets[j];
    const size_t sz = result->chunk_sizes[j];
    if (delta != sz) {
      log_error("contract: contiguous offsets[%" PRIu64 "+1]-offsets[%" PRIu64
                "]=%zu, chunk_sizes[%" PRIu64 "]=%zu",
                j,
                j,
                delta,
                j,
                sz);
      ++violations;
    }
  }
  return violations;
}
