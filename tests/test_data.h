#pragma once

#include "dimension.h"
#include "writer.h"

#include <stddef.h>
#include <stdint.h>

typedef void (*fill_fn)(uint16_t* buf,
                        size_t count,
                        size_t offset,
                        size_t total);

void
fill_zeros(uint16_t* buf, size_t count, size_t offset, size_t total);
void
fill_rand(uint16_t* buf, size_t count, size_t offset, size_t total);
void
fill_xor(uint16_t* buf, size_t count, size_t offset, size_t total);

void
xor_pattern_init(const struct dimension* dims, uint8_t rank, size_t nframes);
void
xor_pattern_free(void);

void
rand_pattern_init(const struct dimension* dims, uint8_t rank, size_t nframes);
void
rand_pattern_free(void);

size_t
fill_pattern_period(fill_fn fill);

size_t
dim_total_elements(const struct dimension* dims, uint8_t rank);

// Distinct contents across blocks keep compression ratios realistic; identical
// contents make compressed-codec results comparable only within one mode.
enum pump_mode
{
  PUMP_CYCLE_BLOCKS = 0,
  PUMP_BUSY_PRODUCER,
  PUMP_SINGLE_BLOCK,
};

#define PUMP_CYCLE_BLOCK_COUNT 8

struct pump_blocks
{
  uint16_t** block;
  size_t count;
  double fill_s;
};

#define PUMP_BLOCK_ELEMENTS (32 * 1024 * 1024)

int
pump_blocks_alloc(struct pump_blocks* out,
                  size_t total_elements,
                  fill_fn fill,
                  size_t bpe,
                  enum pump_mode mode,
                  int measure_fill);

void
pump_blocks_free(struct pump_blocks* b);

// Fill data, pump through writer, flush. Returns 0 on success.
int
pump_data(struct writer* w, size_t total_elements, fill_fn fill);

// Like pump_data but with explicit bytes-per-element.
// Fill still works on uint16_t buffers; the slice end is trimmed to n*bpe.
int
pump_data_bpe(struct writer* w,
              size_t total_elements,
              fill_fn fill,
              size_t bpe);

int
pump_data_modal(struct writer* w,
                size_t total_elements,
                fill_fn fill,
                size_t bpe,
                enum pump_mode mode,
                double* out_fill_s);
