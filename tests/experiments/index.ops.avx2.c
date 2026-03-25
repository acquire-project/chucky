#include "div.avx2.h"
#include "index.ops.util.h"
#include "platform/platform.h"
#include <immintrin.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER
#include <malloc.h>
#define aligned_alloc(align, size) _aligned_malloc((size), (align))
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) free(ptr)
#endif

// AVX2 version of vadd - processes 4 elements at a time
// Specialized version of vadd2_avx2 for step=1
static uint64_t*
vadd_avx2(int rank,
          const int* restrict shape,
          const int* restrict strides,
          uint64_t beg,
          uint64_t end)
{
  // get the output offset corresponding to beg
  uint64_t o = ravel_i32(rank, shape, strides, beg);

  const size_t n_actual = end - beg;
  // Pad n to nearest multiple of 4 for SIMD processing
  const size_t n = ((n_actual + 3) / 4) * 4;

  uint64_t* out = (uint64_t*)aligned_alloc(32, n * sizeof(uint64_t));
  if (!out)
    return 0;

  {
    // For step=1, delta is just the innermost stride
    const uint64_t delta = strides[rank - 1];

    // init out with delta - later we'll apply carry corrections
    const __m256i vdelta = _mm256_set1_epi64x(delta);
    for (size_t i = 0; i < n; i += 4) {
      _mm256_store_si256((__m256i*)&out[i], vdelta);
    }
  }

  // transposed carries - variable-radix arithmetic
  {
    // carry: tracks cumulative carries from dimension d+1
    uint64_t* carry = (uint64_t*)aligned_alloc(32, n * sizeof(uint64_t));
    if (!carry) {
      aligned_free(out);
      return 0;
    }

    for (size_t i = 0; i < n; ++i) {
      carry[i] = 0;
    }

    uint64_t rest_beg = beg;
    uint64_t rest_step = 1;

    for (int d = rank - 1; d > 0; --d) {
      const uint64_t e = rest_beg % shape[d];
      const uint64_t s = rest_step % shape[d];
      rest_beg /= shape[d];
      rest_step /= shape[d];

      const uint64_t correction = strides[d - 1] - shape[d] * strides[d];

      // Compute magic multiplier for division by shape[d]
      uint64_t magic_m;
      int magic_s;
      compute_magic_div(shape[d], &magic_m, &magic_s);

      // Broadcast constants for SIMD
      const __m256i ve = _mm256_set1_epi64x(e);
      const __m256i vs = _mm256_set1_epi64x(s);
      const __m256i vcorrection = _mm256_set1_epi64x(correction);

      // Track previous wraps value for scan operation
      uint64_t prev_wraps = 0;

      // SIMD loop for elements 0..n (32-byte aligned)
      for (size_t i = 0; i + 4 <= n; i += 4) {
        // For step=1: total = e + i*s + carry[i]
        // where s may be 0 or non-zero depending on dimension
        const __m256i vi = _mm256_set_epi64x(i + 3, i + 2, i + 1, i);
        const __m256i vcarry_i = _mm256_load_si256((const __m256i*)&carry[i]);
        const __m256i vi_times_s = mm256_mullo_epi64(vi, vs);
        const __m256i vtotal =
          _mm256_add_epi64(_mm256_add_epi64(ve, vi_times_s), vcarry_i);

        // wraps = total / shape[d] using magic multiplier
        const __m256i vwraps = mm256_div_epu64_const(vtotal, magic_m, magic_s);

        // Compute delta_wraps using scan: wraps[j] - wraps[j-1]
        __m256i vprev = _mm256_set_epi64x(0, 0, 0, prev_wraps);
        __m256i vwraps_shifted = _mm256_permute4x64_epi64(
          vwraps, 0x90); // [wraps[i], wraps[i], wraps[i+1], wraps[i+2]]
        const __m256i vwraps_prev = _mm256_blend_epi32(
          vwraps_shifted,
          vprev,
          0x03); // [prev_wraps, wraps[i], wraps[i+1], wraps[i+2]]

        // delta_wraps = wraps - wraps_prev
        const __m256i vdelta_wraps = _mm256_sub_epi64(vwraps, vwraps_prev);

        // Save wraps[3] for next iteration
        uint64_t wraps_arr[4];
        _mm256_storeu_si256((__m256i*)wraps_arr, vwraps);
        prev_wraps = wraps_arr[3];

        // out[i] += delta_wraps * correction
        const __m256i vcorr_amount =
          mm256_mullo_epi64(vdelta_wraps, vcorrection);
        const __m256i vout = _mm256_load_si256((const __m256i*)&out[i]);
        const __m256i vout_new = _mm256_add_epi64(vout, vcorr_amount);
        _mm256_store_si256((__m256i*)&out[i], vout_new);

        // carry[i] = wraps
        _mm256_store_si256((__m256i*)&carry[i], vwraps);
      }
    }

    aligned_free(carry);
  }

  // prefix sum (only over actual elements, not padding)
  {
    out[0] = o;
    for (size_t i = 1; i < n_actual; ++i)
      out[i] += out[i - 1];
  }
  return out;
}

// AVX2 version of vadd2 - processes 4 elements at a time
static uint64_t*
vadd2_avx2(int rank,
           const int* restrict shape,
           const int* restrict strides,
           uint64_t beg,
           uint64_t end,
           uint64_t step)
{
  // get the output offset corresponding to beg
  uint64_t o = ravel_i32(rank, shape, strides, beg);

  size_t n = (end - beg + step - 1) / step;
  // Pad n to nearest multiple of 4 for SIMD processing
  n = ((n + 3) / 4) * 4;

  uint64_t* out = (uint64_t*)aligned_alloc(32, n * sizeof(uint64_t));
  if (!out)
    return 0;

  {
    // delta: output offset corresponding to a `step` in the input index space
    const uint64_t delta = ravel_i32(rank, shape, strides, step);

    // init out with delta - later we'll apply carry corrections
    const __m256i vdelta = _mm256_set1_epi64x(delta);
    for (size_t i = 0; i < n; i += 4) {
      _mm256_store_si256((__m256i*)&out[i], vdelta);
    }
  }

  // transposed carries - variable-radix arithmetic
  {
    // carry: tracks cumulative carries from dimension d+1
    uint64_t* carry = (uint64_t*)aligned_alloc(32, n * sizeof(uint64_t));
    if (!carry) {
      aligned_free(out);
      return 0;
    }

    for (size_t i = 0; i < n; ++i) {
      carry[i] = 0;
    }

    uint64_t rest_beg = beg;
    uint64_t rest_step = step;

    for (int d = rank - 1; d > 0; --d) {
      const uint64_t e = rest_beg % shape[d];
      const uint64_t s = rest_step % shape[d];
      rest_beg /= shape[d];
      rest_step /= shape[d];

      const uint64_t correction = strides[d - 1] - shape[d] * strides[d];

      // Compute magic multiplier for division by shape[d]
      uint64_t magic_m;
      int magic_s;
      compute_magic_div(shape[d], &magic_m, &magic_s);

      // Broadcast constants for SIMD
      const __m256i ve = _mm256_set1_epi64x(e);
      const __m256i vs = _mm256_set1_epi64x(s);
      const __m256i vcorrection = _mm256_set1_epi64x(correction);

      // Track previous wraps value for scan operation
      uint64_t prev_wraps = 0;

      // SIMD loop for elements 0..n (32-byte aligned)
      for (size_t i = 0; i + 4 <= n; i += 4) {
        // Load i values: [i, i+1, i+2, i+3]
        const __m256i vi = _mm256_set_epi64x(i + 3, i + 2, i + 1, i);

        // Load carry[i] for i, i+1, i+2, i+3
        const __m256i vcarry_i = _mm256_load_si256((const __m256i*)&carry[i]);

        // total = e + i * s + carry[i]
        const __m256i vi_times_s = mm256_mullo_epi64(vi, vs);
        const __m256i vtotal =
          _mm256_add_epi64(_mm256_add_epi64(ve, vi_times_s), vcarry_i);

        // wraps = total / shape[d] using magic multiplier
        // vwraps = [wraps[i], wraps[i+1], wraps[i+2], wraps[i+3]]
        const __m256i vwraps = mm256_div_epu64_const(vtotal, magic_m, magic_s);

        // Compute delta_wraps using scan: wraps[j] - wraps[j-1]
        // We need [prev_wraps, wraps[i], wraps[i+1], wraps[i+2]]
        // Shift vwraps right by 1 element and insert prev_wraps at position 0

        // Use permute to shift: vwraps has lanes [0,1,2,3], we want [prev,
        // 0,1,2] First create a vector with prev_wraps in lane 0
        __m256i vprev = _mm256_set_epi64x(0, 0, 0, prev_wraps);
        // Permute vwraps to get [wraps[2], wraps[1], wraps[0], wraps[0]]
        __m256i vwraps_shifted =
          _mm256_permute4x64_epi64(vwraps, 0x90); // [0,0,1,2]
        // Blend to get [wraps[2], wraps[1], wraps[0], prev_wraps]
        const __m256i vwraps_prev =
          _mm256_blend_epi32(vwraps_shifted, vprev, 0x03);

        // delta_wraps = wraps - wraps_prev
        const __m256i vdelta_wraps = _mm256_sub_epi64(vwraps, vwraps_prev);

        // Save wraps[3] for next iteration
        uint64_t wraps_arr[4];
        _mm256_storeu_si256((__m256i*)wraps_arr, vwraps);
        prev_wraps = wraps_arr[3];

        // out[i] += delta_wraps * correction
        const __m256i vcorr_amount =
          mm256_mullo_epi64(vdelta_wraps, vcorrection);
        const __m256i vout = _mm256_load_si256((const __m256i*)&out[i]);
        const __m256i vout_new = _mm256_add_epi64(vout, vcorr_amount);
        _mm256_store_si256((__m256i*)&out[i], vout_new);

        // carry[i] = wraps
        _mm256_store_si256((__m256i*)&carry[i], vwraps);
      }
    }

    aligned_free(carry);
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
vadd_avx2_agrees_with_add(void)
{
  // Setup a tiling of a 640x480x3 array into chunks of size 10x10x1
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

        const uint64_t beg = (uint64_t)rand() % n;
        const uint64_t* expected =
          make_expected_step(rank, shape, transposed_strides, beg, n, 1);
        if (!expected) {
          state.ecode = 1;
          continue;
        }

        const uint64_t* actual =
          vadd_avx2(rank, shape, transposed_strides, beg, n);
        if (!actual) {
          free((void*)expected);
          state.ecode = 1;
          continue;
        }

        const size_t count = n - beg;
        {
          char buf[64] = { 0 };
          snprintf(buf, sizeof(buf), "beg=%llu", (unsigned long long)beg);
          int err = expect_arrays_equal(expected, actual, count, buf);
          if (err) {
            state.ecode |= err;
          }
        }

        free((void*)expected);
        aligned_free((void*)actual);

        atomic_fetch_add_explicit(&state.completed, 1, memory_order_relaxed);
      }
    }
  }

  return state.ecode;
}

static int
vadd2_avx2_agrees_with_add(void)
{
  // Setup a tiling of a 640x480x3 array into chunks of size 10x10x1
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
#if 1
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
#endif

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
            vadd2_avx2(rank, shape, transposed_strides, beg, n, step);
          if (!actual) {
            free((void*)expected);
            state.ecode = 1;
            continue;
          }

          const size_t count = (n - beg + step - 1) / step;
          {
            char buf[64] = { 0 };
            snprintf(buf,
                     sizeof(buf),
                     "beg=%llu, step=%llu",
                     (unsigned long long)beg,
                     (unsigned long long)step);
            int err = expect_arrays_equal(expected, actual, count, buf);
            if (err) {
              state.ecode |= err;
            }
          }

          free((void*)expected);
          aligned_free((void*)actual);

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
  ecode |= vadd_avx2_agrees_with_add();
  ecode |= vadd2_avx2_agrees_with_add();

  return ecode;
}
