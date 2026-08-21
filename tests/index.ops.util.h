#ifndef INDEX_OPS_UTIL_H
#define INDEX_OPS_UTIL_H

#include "stream/layouts.h"
#include "util/index.ops.h"

#include <stddef.h>
#include <stdint.h>

// Array printing utilities
void
print_vi32(int n, const int* v);

void
println_vi32(int n, const int* v);

void
print_vu64(int n, const uint64_t* v);

void
println_vu64(int n, const uint64_t* v);

void
print_vi64(int n, const int64_t* v);

void
println_vi64(int n, const int64_t* v);

// Test utilities
// Helper to create expected array using ravel_i32()
uint64_t*
make_expected(int rank,
              const int* shape,
              const int* strides,
              uint64_t beg,
              uint64_t end);

// Helper to create expected array using ravel_i32() with step
uint64_t*
make_expected_step(int rank,
                   const int* shape,
                   const int* strides,
                   uint64_t beg,
                   uint64_t end,
                   uint64_t step);

// Compare arrays and report first mismatch
// Returns 0 if arrays match, 1 if they differ
int
expect_arrays_equal(const uint64_t* expected,
                    const uint64_t* actual,
                    size_t n,
                    const char* test_name);

// Generate random test cases
uint64_t*
random_vu64(int count, uint64_t max);

// CPU reference permutation: same unravel-dot logic as the GPU kernel.
uint32_t
cpu_perm(uint64_t i,
         uint8_t lifted_rank,
         const uint64_t* shape,
         const int64_t* strides);

// Chunk alignment for test layouts. The codec's own alignment is a few bytes,
// which pads nothing at these chunk sizes; a wider one keeps the padding a
// real stream has.
#define TEST_CHUNK_ALIGNMENT 4096

// The layout the stream computes for one level, so tests scatter with the
// chunk stride and lifted strides production uses. storage_order may be NULL
// for identity order. Returns 0 on success.
int
test_level_layout(struct tile_stream_layout* out,
                  uint8_t rank,
                  uint8_t n_append,
                  const uint64_t* dim_sizes,
                  const uint64_t* chunk_sizes,
                  const uint8_t* storage_order,
                  size_t bytes_per_element,
                  size_t alignment);

// The pool pads each chunk out to the codec's alignment, so a region is wider
// than an epoch's worth of elements.
#define TEST_REGION_PAD_ELEMENTS 3

// A scatter should use this offset for the element sitting this many elements
// past the start of the first epoch's region.
uint64_t
expected_scatter_offset(uint8_t lifted_rank,
                        const uint64_t* lifted_shape,
                        const int64_t* lifted_strides,
                        uint64_t epoch_elements,
                        uint64_t region_elements,
                        uint64_t offset);

#endif // INDEX_OPS_UTIL_H
