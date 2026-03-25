#ifndef INDEX_OPS_UTIL_H
#define INDEX_OPS_UTIL_H

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

// Build lifted shape and strides for a chunk decomposition (no codec
// alignment). The epoch dimension (dim 0) stride is set to 0 so all epochs
// collapse. storage_order may be NULL for identity order.
void
build_lifted_layout(int rank,
                    const uint64_t* dim_sizes,
                    const uint64_t* chunk_sizes,
                    const uint8_t* storage_order,
                    uint8_t* out_lifted_rank,
                    uint64_t* lifted_shape,
                    int64_t* lifted_strides,
                    uint64_t* out_chunk_elements,
                    uint64_t* out_chunk_stride,
                    uint64_t* out_chunks_per_epoch,
                    uint64_t* out_epoch_elements);

#endif // INDEX_OPS_UTIL_H
