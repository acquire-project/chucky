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
dim_total_elements(const struct dimension* dims, uint8_t rank);

// Regenerate every appended block. Returns 0 on success.
int
pump_data(struct writer* w, size_t total_elements, fill_fn fill);

// Like pump_data but with explicit bytes-per-element. The fill writes
// uint16_t; the slice end is trimmed to n*bpe.
int
pump_data_bpe(struct writer* w,
              size_t total_elements,
              fill_fn fill,
              size_t bpe);

// Fill one block once and reuse it for every append. block_elements is how much
// is handed over per append; 0 uses a default. Pass one frame's worth to see
// what a caller feeding frames would see.
int
pump_data_prefill_blocked(struct writer* w,
                          size_t total_elements,
                          fill_fn fill,
                          size_t bpe,
                          size_t block_elements);
