#include "dimension.h"
#include "defs.limits.h"
#include "util/prelude.h"

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
      .axis_type = dimension_axis_space,
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
  // The append dimension (index 0) must stay at storage position 0.
  if (rank > 0 && order[0] != dims[0].name[0])
    return 1;
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
                       const uint8_t* ratios)
{
  if (nelem == 0)
    return;

  // total_bits = floor(log2(nelem))
  int total_bits = 63 - clzll(nelem);

  uint32_t sum_ratios = 0;
  for (uint8_t i = 0; i < rank; ++i)
    sum_ratios += ratios[i];

  if (sum_ratios == 0)
    return;

  // Greedy bit allocation: each bit goes to the most underserved
  // dimension (lowest bits[i]/ratio[i]). Ties favor higher indices.
  int bits[HALF_MAX_RANK] = {0};
  for (int b = 0; b < total_bits; ++b) {
    int best = -1;
    for (uint8_t i = 0; i < rank; ++i) {
      if (ratios[i] == 0)
        continue;
      if (best < 0 ||
          bits[i] * ratios[best] <= bits[best] * ratios[i])
        best = i;
    }
    bits[best]++;
  }

  for (uint8_t i = 0; i < rank; ++i)
    dims[i].chunk_size = (uint64_t)1 << bits[i];
}

void
dims_budget_chunk_bytes(struct dimension* dims,
                        uint8_t rank,
                        size_t target_chunk_bytes,
                        size_t bytes_per_element,
                        const uint8_t* ratios)
{
  if (bytes_per_element == 0 || target_chunk_bytes < bytes_per_element)
    return;
  dims_budget_chunk_size(dims, rank, target_chunk_bytes / bytes_per_element,
                         ratios);
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
dims_validate(const struct dimension* dims, uint8_t rank)
{
  CHECK(Fail, dims);
  CHECK(Fail, rank > 0 && rank <= HALF_MAX_RANK);

  // chunk_size > 0
  for (int d = 0; d < rank; ++d) {
    if (dims[d].chunk_size == 0) {
      log_error("dims[%d].chunk_size must be > 0", d);
      goto Fail;
    }
  }

  // unbounded dim 0 requires chunks_per_shard > 0
  if (dims[0].size == 0 && dims[0].chunks_per_shard == 0) {
    log_error("dims[0].size=0 (unbounded) requires chunks_per_shard > 0");
    goto Fail;
  }

  // storage_position: valid permutation with dims[0] pinned to 0
  {
    if (dims[0].storage_position != 0) {
      log_error("dims[0].storage_position must be 0");
      goto Fail;
    }
    uint32_t seen = 0;
    for (int d = 0; d < rank; ++d) {
      uint8_t j = dims[d].storage_position;
      if (j >= rank || (seen & (1u << j))) {
        log_error("invalid storage_position permutation at dims[%d]", d);
        goto Fail;
      }
      seen |= (1u << j);
    }
  }

  return 0;
Fail:
  return 1;
}

void
dims_print(const struct dimension* dims, uint8_t rank)
{
  fprintf(stderr, "dim  name  %10s  %10s  %8s  %6s  %8s  storage  ds\n",
         "size",
         "chunk",
         "chunks",
         "cps",
         "shards");
  uint64_t chunk_elements = 1;
  uint64_t chunks_per_epoch = 1;
  for (uint8_t i = 0; i < rank; ++i) {
    uint64_t tc = ceildiv(dims[i].size, dims[i].chunk_size);
    uint64_t cps = dims[i].chunks_per_shard ? dims[i].chunks_per_shard : tc;
    uint64_t sc = ceildiv(tc, cps);
    fprintf(stderr, "%3d  %-4s  %10llu  %10llu  %8llu  %6llu  %8llu  %7d  %s\n",
           i,
           dims[i].name ? dims[i].name : "?",
           (unsigned long long)dims[i].size,
           (unsigned long long)dims[i].chunk_size,
           (unsigned long long)tc,
           (unsigned long long)cps,
           (unsigned long long)sc,
           (int)dims[i].storage_position,
           dims[i].downsample ? "Y" : ".");
    chunk_elements *= dims[i].chunk_size;
    if (i > 0)
      chunks_per_epoch *= tc;
  }
  double epoch_elements = (double)chunks_per_epoch * (double)chunk_elements;
  fprintf(stderr, "chunk_elements: %llu  chunks/epoch: %llu  epoch_elements: %.3g\n",
         (unsigned long long)chunk_elements,
         (unsigned long long)chunks_per_epoch,
         epoch_elements);
}
