#include "dimension.h"
#include "prelude.h"
#include "stream.h"

#include <stdio.h>

#include <string.h>

#ifdef _MSC_VER
#include <intrin.h>
static inline int clzll(unsigned long long x) {
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
      .tile_size = sizes[i],
      .tiles_per_shard = 0,
      .name = name_table[ch],
      .downsample = 0,
      .storage_position = i,
    };
  }
  return rank;
}

int
dims_set_storage_order(struct dimension* dims,
                       uint8_t rank,
                       const char* order)
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
dims_budget_tile_size(struct dimension* dims,
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

  int bits_per_part = (total_bits + (int)sum_ratios - 1) / (int)sum_ratios;
  int remainder = total_bits - bits_per_part * (int)sum_ratios;

  int first_nonzero = -1;
  for (uint8_t i = 0; i < rank; ++i) {
    if (ratios[i] > 0 && first_nonzero < 0)
      first_nonzero = i;
  }

  for (uint8_t i = 0; i < rank; ++i) {
    if (ratios[i] == 0) {
      dims[i].tile_size = 1;
    } else {
      int bits = ratios[i] * bits_per_part;
      if (i == first_nonzero)
        bits += remainder;
      dims[i].tile_size = (uint64_t)1 << bits;
    }
  }
}

void
dims_set_shard_counts(struct dimension* dims,
                      uint8_t rank,
                      const uint64_t* shard_counts)
{
  for (uint8_t i = 0; i < rank; ++i) {
    if (shard_counts[i] == 0)
      continue;
    uint64_t tile_count = ceildiv(dims[i].size, dims[i].tile_size);
    dims[i].tiles_per_shard = ceildiv(tile_count, shard_counts[i]);
  }
}

void
dims_print(const struct dimension* dims, uint8_t rank)
{
  printf("dim  name  %10s  %10s  %8s  %6s  %8s  storage  ds\n",
         "size",
         "tile",
         "tiles",
         "tps",
         "shards");
  uint64_t tile_elements = 1;
  uint64_t tiles_per_epoch = 1;
  for (uint8_t i = 0; i < rank; ++i) {
    uint64_t tc = ceildiv(dims[i].size, dims[i].tile_size);
    uint64_t tps = dims[i].tiles_per_shard ? dims[i].tiles_per_shard : tc;
    uint64_t sc = ceildiv(tc, tps);
    printf("%3d  %-4s  %10llu  %10llu  %8llu  %6llu  %8llu  %7d  %s\n",
           i,
           dims[i].name ? dims[i].name : "?",
           (unsigned long long)dims[i].size,
           (unsigned long long)dims[i].tile_size,
           (unsigned long long)tc,
           (unsigned long long)tps,
           (unsigned long long)sc,
           (int)dims[i].storage_position,
           dims[i].downsample ? "Y" : ".");
    tile_elements *= dims[i].tile_size;
    if (i > 0)
      tiles_per_epoch *= tc;
  }
  double epoch_elements = (double)tiles_per_epoch * (double)tile_elements;
  printf("tile_elements: %llu  tiles/epoch: %llu  epoch_elements: %.3g\n",
         (unsigned long long)tile_elements,
         (unsigned long long)tiles_per_epoch,
         epoch_elements);
}
