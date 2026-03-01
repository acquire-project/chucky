#ifndef INDEX_OPS_UTIL_H
#define INDEX_OPS_UTIL_H

#include "index.ops.h"

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

#endif // INDEX_OPS_UTIL_H
