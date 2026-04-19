#include "dimension.h"
#include "defs.limits.h"
#include "lod/lod_plan.h"
#include "util/prelude.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#ifdef _MSC_VER
#include <intrin.h>
static inline int
clzll(unsigned long long x)
{
  unsigned long index;
  _BitScanReverse64(&index, x);
  return 63 - (int)index;
}
#else
#define clzll(x) __builtin_clzll(x)
#endif

// Static name storage: single-char strings indexed by character value.
// dims[i].name points into this table, avoiding lifetime issues.
static const char name_table[128][2] = {
  ['a'] = "a", ['b'] = "b", ['c'] = "c", ['d'] = "d", ['e'] = "e", ['f'] = "f",
  ['g'] = "g", ['h'] = "h", ['i'] = "i", ['j'] = "j", ['k'] = "k", ['l'] = "l",
  ['m'] = "m", ['n'] = "n", ['o'] = "o", ['p'] = "p", ['q'] = "q", ['r'] = "r",
  ['s'] = "s", ['t'] = "t", ['u'] = "u", ['v'] = "v", ['w'] = "w", ['x'] = "x",
  ['y'] = "y", ['z'] = "z", ['A'] = "A", ['B'] = "B", ['C'] = "C", ['D'] = "D",
  ['E'] = "E", ['F'] = "F", ['G'] = "G", ['H'] = "H", ['I'] = "I", ['J'] = "J",
  ['K'] = "K", ['L'] = "L", ['M'] = "M", ['N'] = "N", ['O'] = "O", ['P'] = "P",
  ['Q'] = "Q", ['R'] = "R", ['S'] = "S", ['T'] = "T", ['U'] = "U", ['V'] = "V",
  ['W'] = "W", ['X'] = "X", ['Y'] = "Y", ['Z'] = "Z", ['0'] = "0", ['1'] = "1",
  ['2'] = "2", ['3'] = "3", ['4'] = "4", ['5'] = "5", ['6'] = "6", ['7'] = "7",
  ['8'] = "8", ['9'] = "9",
};

uint8_t
dims_create(struct dimension* dims, const char* names, const uint64_t* sizes)
{
  if (!dims || !names || !sizes)
    return 0;
  size_t len = strlen(names);
  if (len == 0 || len > HALF_MAX_RANK)
    return 0;
  uint8_t rank = (uint8_t)len;
  for (uint8_t i = 0; i < rank; ++i) {
    unsigned char ch = (unsigned char)names[i];
    if (ch >= 128 || name_table[ch][0] == '\0')
      return 0;
    dims[i] = (struct dimension){
      .size = sizes[i],
      .chunk_size = sizes[i],
      .chunks_per_shard = 0,
      .name = name_table[ch],
      .downsample = 0,
      .storage_position = i,
    };
  }
  return rank;
}

int
dims_set_storage_order(struct dimension* dims, uint8_t rank, const char* order)
{
  if (!order) {
    for (uint8_t i = 0; i < rank; ++i)
      dims[i].storage_position = i;
    return 0;
  }
  if (strlen(order) != rank)
    return 1;
  // Append dimensions must stay at their identity storage positions.
  uint8_t na = dims_n_append(dims, rank);
  for (uint8_t d = 0; d < na; ++d) {
    if (order[d] != dims[d].name[0])
      return 1;
  }
  for (uint8_t i = 0; i < rank; ++i) {
    // Find which dim has name order[i].
    int found = 0;
    for (uint8_t j = 0; j < rank; ++j) {
      if (dims[j].name[0] == order[i]) {
        dims[j].storage_position = i;
        found = 1;
        break;
      }
    }
    if (!found)
      return 1;
  }
  return 0;
}

void
dims_set_downsample_by_name(struct dimension* dims,
                            uint8_t rank,
                            const char* names)
{
  for (uint8_t i = 0; i < rank; ++i)
    dims[i].downsample = (names && strchr(names, dims[i].name[0])) ? 1 : 0;
}

void
dims_set_chunk_sizes(struct dimension* dims,
                     uint8_t rank,
                     const uint64_t* chunk_sizes)
{
  for (uint8_t i = 0; i < rank; ++i)
    dims[i].chunk_size = chunk_sizes[i];
}

void
dims_budget_chunk_size(struct dimension* dims,
                       uint8_t rank,
                       uint64_t nelem,
                       const int* ratios)
{
  if (nelem == 0)
    return;

  // Classify dims. effective[i]:
  //   >0  : bit-budget participant with this weight
  //    0  : chunk_size = 1 (no bits)
  //   -1  : pin at dims[i].size (only for bounded dims; unbounded falls back
  //         to weight=1 so the dim absorbs the remaining budget).
  int effective[HALF_MAX_RANK];
  uint64_t pinned_prod = 1;
  int any_participant = 0;

  for (uint8_t i = 0; i < rank; ++i) {
    if (ratios[i] == -1) {
      if (dims[i].size == 0) {
        effective[i] = 1; // unbounded: absorb remaining budget
        any_participant = 1;
      } else {
        effective[i] = -1; // pin at size
        pinned_prod *= dims[i].size;
      }
    } else if (ratios[i] == 0) {
      effective[i] = 0;
    } else {
      effective[i] = ratios[i];
      any_participant = 1;
    }
  }

  // Apply pins.
  for (uint8_t i = 0; i < rank; ++i)
    if (effective[i] == -1)
      dims[i].chunk_size = dims[i].size;

  if (!any_participant)
    return;

  // Remaining element budget for participants.
  uint64_t remaining = pinned_prod ? nelem / pinned_prod : nelem;
  if (remaining < 1)
    remaining = 1;
  int total_bits = 63 - clzll(remaining);

  // Greedy bit allocation: each bit goes to the most underserved
  // participant (lowest bits[i]/effective[i]). Ties favor higher indices.
  int bits[HALF_MAX_RANK] = { 0 };
  for (int b = 0; b < total_bits; ++b) {
    int best = -1;
    for (uint8_t i = 0; i < rank; ++i) {
      if (effective[i] <= 0)
        continue;
      if (best < 0 || bits[i] * effective[best] <= bits[best] * effective[i])
        best = i;
    }
    bits[best]++;
  }

  for (uint8_t i = 0; i < rank; ++i) {
    if (effective[i] > 0)
      dims[i].chunk_size = (uint64_t)1 << bits[i];
    else if (effective[i] == 0)
      dims[i].chunk_size = 1;
    // effective == -1: already set to size above.
  }
}

void
dims_budget_chunk_bytes(struct dimension* dims,
                        uint8_t rank,
                        size_t target_chunk_bytes,
                        size_t bytes_per_element,
                        const int* ratios)
{
  if (bytes_per_element == 0 || target_chunk_bytes < bytes_per_element)
    return;
  dims_budget_chunk_size(
    dims, rank, target_chunk_bytes / bytes_per_element, ratios);
}

void
dims_set_shard_counts(struct dimension* dims,
                      uint8_t rank,
                      const uint64_t* shard_counts)
{
  for (uint8_t i = 0; i < rank; ++i) {
    if (shard_counts[i] == 0)
      continue;
    uint64_t n_chunks = ceildiv(dims[i].size, dims[i].chunk_size);
    dims[i].chunks_per_shard = ceildiv(n_chunks, shard_counts[i]);
  }
}

int
dims_set_shard_geometry(struct dimension* dims,
                        uint8_t rank,
                        size_t min_shard_bytes,
                        uint32_t target_concurrent_shards,
                        uint32_t min_append_shards,
                        size_t bytes_per_element)
{
  if (!dims || rank == 0 || bytes_per_element == 0)
    return 1;
  for (uint8_t d = 0; d < rank; ++d)
    if (dims[d].chunk_size == 0)
      return 1;

  size_t chunk_bytes = bytes_per_element;
  for (uint8_t d = 0; d < rank; ++d)
    chunk_bytes *= dims[d].chunk_size;

  if (min_shard_bytes > 0 && min_shard_bytes < chunk_bytes) {
    log_error("min_shard_bytes (%zu) is smaller than one chunk (%zu bytes)",
              min_shard_bytes,
              chunk_bytes);
    return 1;
  }

  uint8_t na = dims_n_append(dims, rank);
  uint32_t M = target_concurrent_shards ? target_concurrent_shards : 1;
  const int min_shards_set = min_append_shards > 1;

  uint64_t n_chunks[HALF_MAX_RANK];
  uint64_t shards[HALF_MAX_RANK];
  for (uint8_t d = 0; d < rank; ++d) {
    n_chunks[d] = ceildiv(dims[d].size, dims[d].chunk_size);
    shards[d] = 1;
  }

  // Inner append dims (1..na-1) pass through at their full span — they are
  // part of the parts-budget product but aren't split here.
  uint64_t others_prod = 1;
  for (uint8_t d = 1; d < na; ++d)
    others_prod *= n_chunks[d] ? n_chunks[d] : 1;
  if (others_prod < 1)
    others_prod = 1;

  // Phase A: fill the concurrency target. Integer-greedy (no pow2 waste) —
  // each step grows the inner dim with the largest remaining n_chunks/shards
  // ratio, bounded by Π shards ≤ M.
  uint64_t prod = 1;
  while (prod < (uint64_t)M) {
    int best = -1;
    for (uint8_t d = na; d < rank; ++d) {
      if (shards[d] + 1 > n_chunks[d])
        continue;
      if (prod / shards[d] * (shards[d] + 1) > (uint64_t)M)
        continue;
      if (best < 0 || n_chunks[d] * shards[best] > n_chunks[best] * shards[d])
        best = d;
    }
    if (best < 0)
      break;
    prod = prod / shards[best] * (shards[best] + 1);
    shards[best] += 1;
  }

  uint64_t inner_cps_prod = 1;
  for (uint8_t d = na; d < rank; ++d)
    inner_cps_prod *= ceildiv(n_chunks[d], shards[d]);

  // cps_append_target: the minimum cps_append that the caller's constraints
  // require us to leave room for in the parts budget.
  //   Mode 1 (min_shards_set):  target = floor(n_chunks[0] / min_append_shards)
  //                             — this is the exact cps_append we'll commit to.
  //   Mode 2 (min_shard_bytes): target = cps_floor (depends on inner_cps_prod;
  //                             refreshed inside Phase B as inner shrinks).
  //   Mode 3 (neither):         target = 1.
  uint64_t cps_append_target = 1;
  if (min_shards_set && n_chunks[0] > 0) {
    cps_append_target = n_chunks[0] / min_append_shards;
    if (cps_append_target < 1)
      cps_append_target = 1;
  } else if (min_shard_bytes > 0) {
    uint64_t row_bytes = (uint64_t)chunk_bytes * inner_cps_prod * others_prod;
    if (row_bytes > 0)
      cps_append_target = ceildiv((uint64_t)min_shard_bytes, row_bytes);
    if (cps_append_target < 1)
      cps_append_target = 1;
    if (n_chunks[0] > 0 && cps_append_target > n_chunks[0])
      cps_append_target = n_chunks[0];
  }

  // Phase B: if inner_cps_prod is too big for the parts budget at our target,
  // keep splitting inner dims (past M — target_concurrent_shards is soft).
  // Pick the dim with the largest current cps for the biggest per-split drop.
  // In Mode 2, refresh cps_append_target as inner_cps_prod (and therefore
  // cps_floor) shrinks — this monotonically grows inner_cps_limit.
  for (;;) {
    uint64_t denom = others_prod * cps_append_target;
    if (denom < 1)
      denom = 1;
    uint64_t inner_cps_limit = MAX_PARTS_PER_SHARD / denom;
    if (inner_cps_limit < 1)
      inner_cps_limit = 1;
    if (inner_cps_prod <= inner_cps_limit)
      break;

    // Pick the dim with the largest current cps (biggest per-split absolute
    // reduction) — minimizes final Π shards, keeping us close to the target.
    // Skip dims where ceildiv keeps cps unchanged (integer roundoff stalls).
    int best = -1;
    uint64_t best_cps = 0;
    for (uint8_t d = na; d < rank; ++d) {
      if (shards[d] + 1 > n_chunks[d])
        continue;
      uint64_t cur = ceildiv(n_chunks[d], shards[d]);
      uint64_t nxt = ceildiv(n_chunks[d], shards[d] + 1);
      if (nxt >= cur)
        continue;
      if (cur > best_cps) {
        best = d;
        best_cps = cur;
      }
    }
    if (best < 0)
      break; // no progress possible — can't reduce inner_cps_prod further.
    shards[best] += 1;
    inner_cps_prod = 1;
    for (uint8_t d = na; d < rank; ++d)
      inner_cps_prod *= ceildiv(n_chunks[d], shards[d]);

    if (!min_shards_set && min_shard_bytes > 0) {
      uint64_t row_bytes = (uint64_t)chunk_bytes * inner_cps_prod * others_prod;
      uint64_t t =
        row_bytes ? ceildiv((uint64_t)min_shard_bytes, row_bytes) : 1;
      if (t < 1)
        t = 1;
      if (n_chunks[0] > 0 && t > n_chunks[0])
        t = n_chunks[0];
      cps_append_target = t;
    }
  }

  // Hard failure: if cps_append=1 still exceeds the parts budget after Phase B
  // exhausted all inner splits, the config is genuinely infeasible.
  if (inner_cps_prod * others_prod > MAX_PARTS_PER_SHARD) {
    log_error("cannot satisfy parts budget even at cps_append=1 "
              "(inner_cps_prod=%" PRIu64 ", others_prod=%" PRIu64
              ", MAX_PARTS_PER_SHARD=%d)",
              inner_cps_prod,
              others_prod,
              MAX_PARTS_PER_SHARD);
    return 2;
  }

  // Commit inner cps and inner-append pass-through.
  for (uint8_t d = na; d < rank; ++d)
    dims[d].chunks_per_shard = ceildiv(n_chunks[d], shards[d]);
  for (uint8_t d = 1; d < na; ++d)
    dims[d].chunks_per_shard = n_chunks[d] ? n_chunks[d] : 1;

  // cps_append: maximize within the hard parts cap. min_shard_bytes (soft) is
  // met when Phase B succeeded in reserving enough budget; if it couldn't,
  // we silently accept smaller shards rather than violate the parts cap.
  if (min_shard_bytes > 0) {
    const uint64_t inner_prod = inner_cps_prod * others_prod;
    uint64_t cps_cap = inner_prod ? (MAX_PARTS_PER_SHARD / inner_prod) : 1;
    if (n_chunks[0] > 0 && cps_cap > n_chunks[0])
      cps_cap = n_chunks[0];
    if (min_shards_set && n_chunks[0] > 0) {
      uint64_t cap_for_shards = n_chunks[0] / min_append_shards;
      if (cap_for_shards < 1)
        cap_for_shards = 1;
      if (cps_cap > cap_for_shards)
        cps_cap = cap_for_shards;
    }
    if (cps_cap < 1)
      cps_cap = 1;
    dims[0].chunks_per_shard = cps_cap;
  }

  return 0;
}

int
dims_set_layout(struct dimension* dims,
                uint8_t rank,
                const struct dims_layout_policy* p)
{
  if (!dims || !p || rank == 0)
    return 1;
  if (p->chunk_ratios)
    dims_budget_chunk_bytes(
      dims, rank, p->target_chunk_bytes, p->bytes_per_element, p->chunk_ratios);
  return dims_set_shard_geometry(dims,
                                 rank,
                                 p->min_shard_bytes,
                                 p->target_concurrent_shards,
                                 p->min_append_shards,
                                 p->bytes_per_element);
}
