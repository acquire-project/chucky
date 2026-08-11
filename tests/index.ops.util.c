#include "index.ops.util.h"
#include "defs.limits.h"
#include "stream/config.h"
#include "util/prelude.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void
print_vi32(int n, const int* v)
{
  putc('[', stdout);
  if (n)
    printf("%d", v[0]);
  for (int i = 1; i < n; ++i)
    printf(", %d", v[i]);
  putc(']', stdout);
}

void
println_vi32(int n, const int* v)
{
  print_vi32(n, v);
  putc('\n', stdout);
}

void
print_vu64(int n, const uint64_t* v)
{
  putc('[', stdout);
  if (n)
    printf("%llu", (unsigned long long)v[0]);
  for (int i = 1; i < n; ++i)
    printf(", %llu", (unsigned long long)v[i]);
  putc(']', stdout);
}

void
println_vu64(int n, const uint64_t* v)
{
  print_vu64(n, v);
  putc('\n', stdout);
}

void
print_vi64(int n, const int64_t* v)
{
  putc('[', stdout);
  if (n)
    printf("%lld", (long long)v[0]);
  for (int i = 1; i < n; ++i)
    printf(", %lld", (long long)v[i]);
  putc(']', stdout);
}

void
println_vi64(int n, const int64_t* v)
{
  print_vi64(n, v);
  putc('\n', stdout);
}

// Helper to create expected array using ravel_i32()
uint64_t*
make_expected(int rank,
              const int* shape,
              const int* strides,
              uint64_t beg,
              uint64_t end)
{
  const size_t n = end - beg;
  uint64_t* out = (uint64_t*)malloc(n * sizeof(uint64_t));
  if (!out)
    return 0;

  for (uint64_t i = 0; i < n; ++i) {
    out[i] = ravel_i32(rank, shape, strides, beg + i);
  }
  return out;
}

// Helper to create expected array using ravel_i32() with step
uint64_t*
make_expected_step(int rank,
                   const int* shape,
                   const int* strides,
                   uint64_t beg,
                   uint64_t end,
                   uint64_t step)
{
  const size_t n = (end - beg + step - 1) / step;
  uint64_t* out = (uint64_t*)malloc(n * sizeof(uint64_t));
  if (!out)
    return 0;

  for (uint64_t i = 0; i < n; ++i) {
    out[i] = ravel_i32(rank, shape, strides, beg + i * step);
  }
  return out;
}

// Compare arrays and report first mismatch
// Returns 0 if arrays match, 1 if they differ
int
expect_arrays_equal(const uint64_t* expected,
                    const uint64_t* actual,
                    size_t n,
                    const char* test_name)
{
  for (size_t i = 0; i < n; ++i) {
    if (expected[i] != actual[i]) {
      printf("%s: Expected %llu but got %llu at element %zu\n",
             test_name,
             (unsigned long long)expected[i],
             (unsigned long long)actual[i],
             i);
      return 1;
    }
  }
  return 0;
}

uint64_t*
random_vu64(int count, uint64_t max)
{
  uint64_t* cases = (uint64_t*)malloc(count * sizeof(uint64_t));
  if (!cases)
    return 0;
  for (int i = 0; i < count; ++i) {
    cases[i] = (uint64_t)rand() % max;
  }
  return cases;
}

int
test_level_layout(struct tile_stream_layout* out,
                  uint8_t rank,
                  uint8_t n_append,
                  const uint64_t* dim_sizes,
                  const uint64_t* chunk_sizes,
                  const uint8_t* storage_order,
                  size_t bytes_per_element,
                  size_t alignment)
{
  struct dimension dims[HALF_MAX_RANK] = { 0 };
  uint8_t identity[HALF_MAX_RANK];

  for (uint8_t d = 0; d < rank; ++d) {
    dims[d].size = dim_sizes[d];
    dims[d].chunk_size = chunk_sizes[d];
    identity[d] = d;
  }

  memset(out, 0, sizeof(*out));
  return compute_level_layout(out,
                              rank,
                              n_append,
                              bytes_per_element,
                              dims,
                              dim_sizes,
                              alignment,
                              storage_order ? storage_order : identity);
}

uint64_t
expected_scatter_offset(uint8_t lifted_rank,
                        const uint64_t* lifted_shape,
                        const int64_t* lifted_strides,
                        uint64_t epoch_elements,
                        uint64_t region_elements,
                        uint64_t offset)
{
  return (offset / epoch_elements) * region_elements +
         ravel(
           lifted_rank, lifted_shape, lifted_strides, offset % epoch_elements);
}

uint32_t
cpu_perm(uint64_t i,
         uint8_t lifted_rank,
         const uint64_t* shape,
         const int64_t* strides)
{
  uint64_t out = 0;
  uint64_t rest = i;
  for (int d = lifted_rank - 1; d >= 0; --d) {
    uint64_t coord = rest % shape[d];
    rest /= shape[d];
    out += coord * (uint64_t)strides[d];
  }
  return (uint32_t)out;
}
