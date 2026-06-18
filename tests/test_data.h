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

// Period (in elements) of the active fill pattern; 0 if it has none (zeros).
size_t
fill_pattern_period(fill_fn fill);

size_t
dim_total_elements(const struct dimension* dims, uint8_t rank);

// How the pump produces the data it appends. The choice decides whether the
// harness or the library dominates the measured wall time.
enum pump_mode
{
  // Pre-generate a handful of distinct blocks once, then cycle through them
  // one per append with no in-loop generation. Distinct contents keep
  // compression ratios realistic; per-append cost is just the writer. Default.
  PUMP_CYCLE_BLOCKS = 0,
  // Regenerate the block on every append (the original behaviour). The fill
  // thread competes with the library for host memory bandwidth -- use this to
  // measure the library under a busy producer.
  PUMP_BUSY_PRODUCER,
  // Fill one block once and append it repeatedly. Cheapest, but every block has
  // identical contents, so compressed-codec results are only comparable within
  // this mode (within-mode A/B), not against the varied-data modes.
  PUMP_SINGLE_BLOCK,
};

// Number of distinct blocks pre-generated and cycled by PUMP_CYCLE_BLOCKS.
// Capped to the number of appends for small runs.
#define PUMP_CYCLE_BLOCK_COUNT 8

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

// Pump in the given mode. When out_fill_s is non-NULL, the seconds spent
// generating data (block pre-generation for PUMP_CYCLE_BLOCKS, per-append fill
// for PUMP_BUSY_PRODUCER) are written there so the report can attribute harness
// time. Returns 0 on success.
int
pump_data_modal(struct writer* w,
                size_t total_elements,
                fill_fn fill,
                size_t bpe,
                enum pump_mode mode,
                double* out_fill_s);
