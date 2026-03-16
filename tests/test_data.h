#pragma once

#include "stream.h"

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

// Fill data, pump through writer, flush. Returns 0 on success.
int
pump_data(struct writer* w, size_t total_elements, fill_fn fill);
