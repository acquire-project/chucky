#include "index.ops.util.h"
#include "platform.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// # vadd
// Return the output offsets corresponding to input offsets beg..end
static uint64_t*
vadd(int rank,
     const int* restrict shape,
     const int* restrict strides,
     uint64_t beg,
     uint64_t end)
{
  // get the output offset corresponding to beg
  uint64_t o = ravel_i32(rank, shape, strides, beg);

  const size_t n = end - beg;
  uint64_t* out = (uint64_t*)malloc((end - beg) * sizeof(uint64_t));
  if (!out)
    return 0;

  // init out with deltas - later we'll do a scan
  for (size_t i = 0; i < n; ++i) {
    out[i] = strides[rank - 1];
  }

  // transposed carries
  {
    uint64_t rest = beg;
    int input_stride = 1; // Start with the stride for the last dimension
    int first_carry = (beg % shape[rank - 1]);
    first_carry = (shape[rank - 1] - first_carry);

    for (int d = rank - 1; d > 0; --d) {
      const int e = shape[d];
      rest /= e;
      // Q: Over the next n elements, where will there be a carry from d to d-1?
      // A: Every input_stride*shape[d] elements starting at first_carry.
      const int next_input_stride = input_stride * shape[d];
      // correct for carry from d to d-1
      const uint64_t correction = strides[d - 1] - shape[d] * strides[d];
      for (size_t i = first_carry; i < n; i += next_input_stride) {
        out[i] += correction;
      }

      // Compute first_carry for the next dimension (d - 1)
      if (d > 1) {
        const int r_next = rest % shape[d - 1];
        first_carry =
          first_carry + next_input_stride * (shape[d - 1] - r_next - 1);
      }

      input_stride = next_input_stride;
    }
  }

  // prefix sum
  {
    out[0] = o;
    for (size_t i = 1; i < n; ++i)
      out[i] += out[i - 1];
  }
  return out;
}

// # vadd2
// Return the output offsets corresponding to input offsets beg..end
// stepping by step.
//
// The strides array (indexed by input dimensions) tells us the output offset
// change for a +1 move in each input dimension.
//
// e.g. For a transpose from 5x7x3->3x5x7, strides should be `{7,1,35}`.
static uint64_t*
vadd2(int rank,
      const int* restrict shape,
      const int* restrict strides,
      uint64_t beg,
      uint64_t end,
      uint64_t step)
{
  // get the output offset corresponding to beg
  uint64_t o = ravel_i32(rank, shape, strides, beg);

  const size_t n = (end - beg + step - 1) / step;
  uint64_t* out = (uint64_t*)malloc(n * sizeof(uint64_t));
  if (!out)
    return 0;

  {
    // delta: output offset corresponding to a `step` in the input index space
    const uint64_t delta = ravel_i32(rank, shape, strides, step);

    // init out with delta - later we'll apply carry corrections
    for (size_t i = 0; i < n; ++i) {
      out[i] = delta;
    }
  }

  // transposed carries - variable-radix arithmetic
  {
    // carry: tracks cumulative carries from dimension d+1
    uint64_t* carry = (uint64_t*)calloc(n, sizeof(uint64_t));
    if (!carry) {
      free(out);
      return 0;
    }

    uint64_t rest_beg = beg;
    uint64_t rest_step = step;

    for (int d = rank - 1; d > 0; --d) {
      const uint64_t e = rest_beg % shape[d];
      const uint64_t s = rest_step % shape[d];
      rest_beg /= shape[d];
      rest_step /= shape[d];

      const uint64_t correction = strides[d - 1] - shape[d] * strides[d];

      for (size_t i = 0; i < n; ++i) {
        // Compute current coordinate including incoming carry
        const uint64_t total = e + i * s + carry[i];

        // Count cumulative wraps in this dimension
        const uint64_t wraps = total / shape[d];

        // Compute delta (new wraps at this step)
        const uint64_t delta_wraps = wraps - (i > 0 ? carry[i - 1] : 0);

        // Apply correction for each wrap
        out[i] += delta_wraps * correction;

        // Store cumulative wraps for next dimension
        carry[i] = wraps;
      }
    }

    free(carry);
  }

  // prefix sum
  {
    out[0] = o;
    for (size_t i = 1; i < n; ++i)
      out[i] += out[i - 1];
  }
  return out;
}

static int
vadd_agrees_with_add(void)
{
  // Setup a tiling of a 640x480x3 array into tiles of size 10x10x1
  const int rank = 6;
  const int shape[] = { 48, 10, 64, 10, 3, 1 };
  int transposed_strides[6] = { 0 };
  {
    int transposed_shape[6] = { 0 };
    {
      const int p[] = { 0, 2, 4, 1, 3, 5 };
      permute_i32(rank, p, shape, transposed_shape);
    }
    compute_strides(rank, transposed_shape, transposed_strides);
    printf("%30s", "Input shape: ");
    println_vi32(rank, shape);
    {
      printf("%30s", "Input strides (row-major): ");
      int strides[6] = { 0 };
      compute_strides(rank, shape, strides);
      println_vi32(rank, strides);
    }
    printf("%30s", "Transposed shape: ");
    println_vi32(rank, transposed_shape);
    printf("%30s", "Transposed strides: ");
    println_vi32(rank, transposed_strides);
  }

  const int n = transposed_strides[0] * shape[0];
  const int num_tests = 1000;

  const uint64_t* expected_all =
    make_expected(rank, shape, transposed_strides, 0, n);
  if (!expected_all)
    return 1;

  uint64_t* test_cases = random_vu64(num_tests, n);
  if (!test_cases) {
    free((void*)expected_all);
    return 1;
  }

  struct
  {
    _Atomic int completed;
    _Atomic int ecode;
  } state = { 0 };

#pragma omp parallel
  {
#pragma omp master
    {
      struct platform_clock clk = { 0 };
      platform_toc(&clk);
      int last_completed = 0;

      while (state.completed < num_tests) {
        float dt = platform_toc(&clk);
        int delta = state.completed - last_completed;
        float velocity = dt > 0 ? delta / dt : 0;
        last_completed = state.completed;

        printf("\rProgress: %d/%d (%.1f%%) - %.0f/s",
               state.completed,
               num_tests,
               100.0 * state.completed / num_tests,
               velocity);
        fflush(stdout);
        platform_sleep_ns(500000000LL);
      }
      printf("\n");
    }

    {
      int i;
#pragma omp for schedule(guided)
      for (i = 0; i < num_tests; ++i) {
        if (state.ecode)
          continue;

        const uint64_t beg = test_cases[i];
        const uint64_t* actual = vadd(rank, shape, transposed_strides, beg, n);
        if (!actual) {
          state.ecode = 1;
          continue;
        }

        const uint64_t* expected = expected_all + beg;

        {
          char buf[64] = { 0 };
          snprintf(buf, sizeof(buf), "beg=%llu", (unsigned long long)beg);
          int err = expect_arrays_equal(expected, actual, n - beg, buf);
          if (err) {
            state.ecode |= err;
          }
        }

        free((void*)actual);

        atomic_fetch_add_explicit(&state.completed, 1, memory_order_relaxed);
      }
    }
  }

  free(test_cases);
  free((void*)expected_all);
  return state.ecode;
}

static int
vadd2_agrees_with_add(void)
{
  // Setup a tiling of a 640x480x3 array into tiles of size 10x10x1
  const int rank = 6;
  const int shape[] = { 48, 10, 64, 10, 3, 1 };
  int transposed_strides[6] = { 0 };
  {
    int transposed_shape[6] = { 0 };
    {
      const int p[] = { 0, 2, 4, 1, 3, 5 };
      permute_i32(rank, p, shape, transposed_shape);
    }
    compute_strides(rank, transposed_shape, transposed_strides);
    printf("%30s", "Input shape: ");
    println_vi32(rank, shape);
    {
      printf("%30s", "Input strides (row-major): ");
      int strides[6] = { 0 };
      compute_strides(rank, shape, strides);
      println_vi32(rank, strides);
    }
    printf("%30s", "Transposed shape: ");
    println_vi32(rank, transposed_shape);
    printf("%30s", "Transposed strides: ");
    println_vi32(rank, transposed_strides);
  }

  const int n = transposed_strides[0] * shape[0];
  const int num_tests = 1000;
  const uint64_t steps[] = { 1, 2, 3, 5, 7, 11, 32, 100, 1000 };
  const int num_steps = sizeof(steps) / sizeof(steps[0]);

  struct
  {
    _Atomic int completed;
    _Atomic int ecode;
  } state = { 0 };

#pragma omp parallel
  {
#pragma omp master
    {
      struct platform_clock clk = { 0 };
      platform_toc(&clk);
      int last_completed = 0;
      const int total_tests = num_tests * num_steps;

      while (state.completed < total_tests) {
        float dt = platform_toc(&clk);
        int delta = state.completed - last_completed;
        float velocity = dt > 0 ? delta / dt : 0;
        last_completed = state.completed;

        printf("\rProgress: %d/%d (%.1f%%) - %.0f/s",
               state.completed,
               total_tests,
               100.0 * state.completed / total_tests,
               velocity);
        fflush(stdout);
        platform_sleep_ns(500000000LL);
      }
      printf("\n");
    }

    {
      int step_idx, i;
#pragma omp for schedule(guided) collapse(2)
      for (step_idx = 0; step_idx < num_steps; ++step_idx) {
        for (i = 0; i < num_tests; ++i) {
          if (state.ecode)
            continue;

          const uint64_t step = steps[step_idx];
          const uint64_t beg = (uint64_t)rand() % n;
          const uint64_t* expected =
            make_expected_step(rank, shape, transposed_strides, beg, n, step);
          if (!expected) {
            state.ecode = 1;
            continue;
          }

          const uint64_t* actual =
            vadd2(rank, shape, transposed_strides, beg, n, step);
          if (!actual) {
            free((void*)expected);
            state.ecode = 1;
            continue;
          }

          const size_t count = (n - beg + step - 1) / step;
          {
            char buf[64] = { 0 };
            snprintf(buf, sizeof(buf), "beg=%llu, step=%llu", (unsigned long long)beg, (unsigned long long)step);
            int err = expect_arrays_equal(expected, actual, count, buf);
            if (err) {
              int show = count < 10 ? (int)count : 10;
              printf("Expected (first 10): ");
              println_vu64(show, expected);
              printf("Actual (first 10): ");
              println_vu64(show, actual);
              state.ecode |= err;
            }
          }

          free((void*)expected);
          free((void*)actual);

          atomic_fetch_add_explicit(&state.completed, 1, memory_order_relaxed);
        }
      }
    }
  }

  return state.ecode;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  srand(42);

  int ecode = 0;

  ecode |= vadd_agrees_with_add();
  ecode |= vadd2_agrees_with_add();

  return ecode;
}
