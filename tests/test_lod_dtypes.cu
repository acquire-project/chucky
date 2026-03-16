// Test lod_reduce and lod_accum_fold_fused + lod_accum_emit for all new dtypes.
// Uses C++ templates to avoid per-type boilerplate.

#include "lod.h"
#include "prelude.cuda.h"
#include "prelude.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

// Widened accumulator for lod_reduce (register-only).
template<typename T> struct reduce_acc { using type = T; };
template<> struct reduce_acc<uint8_t>  { using type = uint32_t; };
template<> struct reduce_acc<uint16_t> { using type = uint32_t; };
template<> struct reduce_acc<uint32_t> { using type = uint64_t; };
template<> struct reduce_acc<int8_t>   { using type = int32_t; };
template<> struct reduce_acc<int16_t>  { using type = int32_t; };
template<> struct reduce_acc<int32_t>  { using type = int64_t; };
template<> struct reduce_acc<__half>   { using type = float; };

static int
upload(CUdeviceptr* d, const void* h, size_t bytes)
{
  if (cuMemAlloc(d, bytes) != CUDA_SUCCESS)
    return 0;
  if (cuMemcpyHtoD(*d, h, bytes) != CUDA_SUCCESS) {
    cuMemFree(*d);
    *d = 0;
    return 0;
  }
  return 1;
}

// ---------------------------------------------------------------------------
// Direct lod_reduce test: 8 src elements → 2 dst elements (windows of 4).
// ---------------------------------------------------------------------------

template<typename T>
static int
test_reduce(const char* label,
            enum lod_dtype dtype,
            enum lod_reduce_method method)
{
  using Acc = typename reduce_acc<T>::type;
  log_info("=== %s ===", label);
  int ok = 0;
  CUstream stream = NULL;
  CUdeviceptr d_values = 0, d_ends = 0;

  T src[8];
  for (int i = 0; i < 8; ++i)
    src[i] = (T)(i + 1);

  // CPU reference
  T expected[2];
  for (int w = 0; w < 2; ++w) {
    T* win = src + w * 4;
    if (method == lod_reduce_mean) {
      Acc sum = (Acc)0;
      for (int j = 0; j < 4; ++j)
        sum += (Acc)win[j];
      expected[w] = (T)(sum / (Acc)4);
    } else if (method == lod_reduce_min) {
      T best = win[0];
      for (int j = 1; j < 4; ++j)
        if (win[j] < best)
          best = win[j];
      expected[w] = best;
    } else if (method == lod_reduce_max) {
      T best = win[0];
      for (int j = 1; j < 4; ++j)
        if (win[j] > best)
          best = win[j];
      expected[w] = best;
    }
  }

  // ends[i] = cumulative window boundary: windows [0,4) and [4,8)
  uint64_t ends[2] = { 4, 8 };

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  // Layout: [8 src][2 dst] in one buffer
  CU(Fail, cuMemAlloc(&d_values, 10 * sizeof(T)));
  CU(Fail, cuMemcpyHtoD(d_values, src, 8 * sizeof(T)));

  CHECK(Fail, upload(&d_ends, ends, 2 * sizeof(uint64_t)));

  CHECK(Fail,
        lod_reduce(d_values, d_ends, dtype, method,
                   0, 8, 8, 2, 1, stream) == 0);
  CU(Fail, cuStreamSynchronize(stream));

  {
    T result[2];
    CU(Fail, cuMemcpyDtoH(result, d_values + 8 * sizeof(T), 2 * sizeof(T)));
    for (int i = 0; i < 2; ++i) {
      double diff = (double)result[i] - (double)expected[i];
      if (diff < 0)
        diff = -diff;
      double eps = sizeof(T) <= 4 ? 1e-5 : 1e-12;
      if (diff > eps) {
        log_error("  FAIL at %d: gpu=%g expected=%g",
                  i, (double)result[i], (double)expected[i]);
        goto Fail;
      }
    }
  }

  log_info("  PASS");
  ok = 1;
Fail:
  cuMemFree(d_values);
  cuMemFree(d_ends);
  cuStreamDestroy(stream);
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Fold/emit test: accumulate n_epochs of data, emit, compare to CPU.
// ---------------------------------------------------------------------------

template<typename T>
static int
test_fold(const char* label,
          enum lod_dtype dtype,
          enum lod_reduce_method method,
          int n_epochs)
{
  log_info("=== %s ===", label);
  int ok = 0;
  const uint64_t N = 64;
  const int nlod = 2;
  size_t accum_bpe = lod_accum_bpe(dtype, method);

  CUstream stream = NULL;
  CUdeviceptr d_accum = 0, d_data = 0, d_out = 0;
  CUdeviceptr d_level_ids = 0, d_counts = 0;

  const int level = 1; // matches memset(h_ids, 1, N) below

  T* h_data = (T*)malloc(n_epochs * N * sizeof(T));
  T* h_result = (T*)malloc(N * sizeof(T));
  T* h_expected = (T*)malloc(N * sizeof(T));
  uint8_t* h_ids = (uint8_t*)malloc(N);
  CHECK(Fail, h_data && h_result && h_expected && h_ids);

  for (int e = 0; e < n_epochs; ++e)
    for (uint64_t i = 0; i < N; ++i)
      h_data[e * N + i] = (T)((e * 10 + i) % 120 + 1);

  for (uint64_t i = 0; i < N; ++i) {
    if (method == lod_reduce_mean) {
      if constexpr (std::is_floating_point_v<T>) {
        // Float: fold sums, emit divides
        T sum = (T)0;
        for (int e = 0; e < n_epochs; ++e)
          sum += h_data[e * N + i];
        h_expected[i] = sum / (T)n_epochs;
      } else {
        // Integer: fold applies overflow_safe_add_shift per step
        T accum = h_data[i];
        int s = level;
        T mask = (T)((1u << s) - 1);
        for (int e = 1; e < n_epochs; ++e) {
          T b = h_data[e * N + i];
          if constexpr (sizeof(T) >= 8)
            accum = accum + b; // 64-bit: raw sum
          else
            accum = (T)((accum >> s) + (b >> s)
                        + (((accum & mask) + (b & mask)) >> s));
        }
        h_expected[i] = accum; // already divided
      }
    } else if (method == lod_reduce_min) {
      T best = h_data[i];
      for (int e = 1; e < n_epochs; ++e)
        if (h_data[e * N + i] < best)
          best = h_data[e * N + i];
      h_expected[i] = best;
    } else if (method == lod_reduce_max) {
      T best = h_data[i];
      for (int e = 1; e < n_epochs; ++e)
        if (h_data[e * N + i] > best)
          best = h_data[e * N + i];
      h_expected[i] = best;
    }
  }

  memset(h_ids, 1, N);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  CU(Fail, cuMemAlloc(&d_accum, N * accum_bpe));
  CU(Fail, cuMemAlloc(&d_data, N * sizeof(T)));
  CU(Fail, cuMemAlloc(&d_out, N * sizeof(T)));
  CHECK(Fail, upload(&d_level_ids, h_ids, N));
  CU(Fail, cuMemAlloc(&d_counts, nlod * sizeof(uint32_t)));

  {
    uint32_t counts[2] = { 0, 0 };
    for (int e = 0; e < n_epochs; ++e) {
      CU(Fail, cuMemcpyHtoD(d_data, h_data + e * N, N * sizeof(T)));
      CU(Fail, cuMemcpyHtoD(d_counts, counts, nlod * sizeof(uint32_t)));
      CHECK(Fail,
            lod_accum_fold_fused(d_accum, d_data, d_level_ids, d_counts,
                                 dtype, method, N, stream) == 0);
      counts[1]++;
    }
  }

  CHECK(Fail,
        lod_accum_emit(d_out, d_accum, dtype, method, N,
                        (uint32_t)n_epochs, stream) == 0);
  CU(Fail, cuStreamSynchronize(stream));
  CU(Fail, cuMemcpyDtoH(h_result, d_out, N * sizeof(T)));

  for (uint64_t i = 0; i < N; ++i) {
    double diff = (double)h_result[i] - (double)h_expected[i];
    if (diff < 0)
      diff = -diff;
    double eps = sizeof(T) <= 4 ? 1e-5 : 1e-12;
    if (diff > eps) {
      log_error("  FAIL at %llu: gpu=%g expected=%g",
                (unsigned long long)i,
                (double)h_result[i], (double)h_expected[i]);
      goto Fail;
    }
  }

  log_info("  PASS");
  ok = 1;
Fail:
  free(h_data);
  free(h_result);
  free(h_expected);
  free(h_ids);
  cuMemFree(d_accum);
  cuMemFree(d_data);
  cuMemFree(d_out);
  cuMemFree(d_level_ids);
  cuMemFree(d_counts);
  cuStreamDestroy(stream);
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// f16 tests: __half has no host-side arithmetic, so use float for CPU ref.
// ---------------------------------------------------------------------------

static int
test_reduce_f16(const char* label, enum lod_reduce_method method)
{
  log_info("=== %s ===", label);
  int ok = 0;
  CUstream stream = NULL;
  CUdeviceptr d_values = 0, d_ends = 0;

  __half src[8];
  for (int i = 0; i < 8; ++i)
    src[i] = __float2half((float)(i + 1));

  // CPU reference in float
  float expected_f[2];
  for (int w = 0; w < 2; ++w) {
    float win[4];
    for (int j = 0; j < 4; ++j)
      win[j] = __half2float(src[w * 4 + j]);
    if (method == lod_reduce_mean) {
      float sum = 0;
      for (int j = 0; j < 4; ++j)
        sum += win[j];
      expected_f[w] = sum / 4.0f;
    } else if (method == lod_reduce_min) {
      float best = win[0];
      for (int j = 1; j < 4; ++j)
        if (win[j] < best)
          best = win[j];
      expected_f[w] = best;
    } else if (method == lod_reduce_max) {
      float best = win[0];
      for (int j = 1; j < 4; ++j)
        if (win[j] > best)
          best = win[j];
      expected_f[w] = best;
    }
  }

  uint64_t ends[2] = { 4, 8 };

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  CU(Fail, cuMemAlloc(&d_values, 10 * sizeof(__half)));
  CU(Fail, cuMemcpyHtoD(d_values, src, 8 * sizeof(__half)));
  CHECK(Fail, upload(&d_ends, ends, 2 * sizeof(uint64_t)));

  CHECK(Fail,
        lod_reduce(d_values, d_ends, lod_dtype_f16, method,
                   0, 8, 8, 2, 1, stream) == 0);
  CU(Fail, cuStreamSynchronize(stream));

  {
    __half result[2];
    CU(Fail,
       cuMemcpyDtoH(result, d_values + 8 * sizeof(__half), 2 * sizeof(__half)));
    for (int i = 0; i < 2; ++i) {
      float got = __half2float(result[i]);
      float diff = fabsf(got - expected_f[i]);
      if (diff > 0.05f) {
        log_error("  FAIL at %d: gpu=%g expected=%g", i, got, expected_f[i]);
        goto Fail;
      }
    }
  }

  log_info("  PASS");
  ok = 1;
Fail:
  cuMemFree(d_values);
  cuMemFree(d_ends);
  cuStreamDestroy(stream);
  return ok ? 0 : 1;
}

static int
test_fold_f16(const char* label,
              enum lod_reduce_method method,
              int n_epochs)
{
  log_info("=== %s ===", label);
  int ok = 0;
  const uint64_t N = 64;
  const int nlod = 2;
  size_t accum_bpe = lod_accum_bpe(lod_dtype_f16, method);

  CUstream stream = NULL;
  CUdeviceptr d_accum = 0, d_data = 0, d_out = 0;
  CUdeviceptr d_level_ids = 0, d_counts = 0;

  __half* h_data = (__half*)malloc(n_epochs * N * sizeof(__half));
  __half* h_result = (__half*)malloc(N * sizeof(__half));
  float* h_expected = (float*)malloc(N * sizeof(float));
  uint8_t* h_ids = (uint8_t*)malloc(N);
  CHECK(Fail, h_data && h_result && h_expected && h_ids);

  for (int e = 0; e < n_epochs; ++e)
    for (uint64_t i = 0; i < N; ++i)
      h_data[e * N + i] = __float2half((float)((e * 10 + i) % 120 + 1));

  for (uint64_t i = 0; i < N; ++i) {
    if (method == lod_reduce_mean) {
      float sum = 0;
      for (int e = 0; e < n_epochs; ++e)
        sum += __half2float(h_data[e * N + i]);
      h_expected[i] = sum / (float)n_epochs;
    } else if (method == lod_reduce_min) {
      float best = __half2float(h_data[i]);
      for (int e = 1; e < n_epochs; ++e) {
        float v = __half2float(h_data[e * N + i]);
        if (v < best)
          best = v;
      }
      h_expected[i] = best;
    } else if (method == lod_reduce_max) {
      float best = __half2float(h_data[i]);
      for (int e = 1; e < n_epochs; ++e) {
        float v = __half2float(h_data[e * N + i]);
        if (v > best)
          best = v;
      }
      h_expected[i] = best;
    }
  }

  memset(h_ids, 1, N);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  CU(Fail, cuMemAlloc(&d_accum, N * accum_bpe));
  CU(Fail, cuMemAlloc(&d_data, N * sizeof(__half)));
  CU(Fail, cuMemAlloc(&d_out, N * sizeof(__half)));
  CHECK(Fail, upload(&d_level_ids, h_ids, N));
  CU(Fail, cuMemAlloc(&d_counts, nlod * sizeof(uint32_t)));

  {
    uint32_t counts[2] = { 0, 0 };
    for (int e = 0; e < n_epochs; ++e) {
      CU(Fail, cuMemcpyHtoD(d_data, h_data + e * N, N * sizeof(__half)));
      CU(Fail, cuMemcpyHtoD(d_counts, counts, nlod * sizeof(uint32_t)));
      CHECK(Fail,
            lod_accum_fold_fused(d_accum, d_data, d_level_ids, d_counts,
                                 lod_dtype_f16, method, N, stream) == 0);
      counts[1]++;
    }
  }

  CHECK(Fail,
        lod_accum_emit(d_out, d_accum, lod_dtype_f16, method, N,
                        (uint32_t)n_epochs, stream) == 0);
  CU(Fail, cuStreamSynchronize(stream));
  CU(Fail, cuMemcpyDtoH(h_result, d_out, N * sizeof(__half)));

  for (uint64_t i = 0; i < N; ++i) {
    float got = __half2float(h_result[i]);
    float diff = fabsf(got - h_expected[i]);
    if (diff > 0.1f) { // f16 has ~3 decimal digits of precision
      log_error("  FAIL at %llu: gpu=%g expected=%g",
                (unsigned long long)i, got, h_expected[i]);
      goto Fail;
    }
  }

  log_info("  PASS");
  ok = 1;
Fail:
  free(h_data);
  free(h_result);
  free(h_expected);
  free(h_ids);
  cuMemFree(d_accum);
  cuMemFree(d_data);
  cuMemFree(d_out);
  cuMemFree(d_level_ids);
  cuMemFree(d_counts);
  cuStreamDestroy(stream);
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------

int
main(void)
{
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
      cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) {
    log_error("CUDA init failed");
    return 1;
  }

  int nfail = 0;

  // --- lod_reduce: mean, min, max for each new dtype ---

#define R(label, T, dtype, method)                                             \
  nfail += test_reduce<T>(label, dtype, method)

  R("reduce_u8_mean",  uint8_t,  lod_dtype_u8,  lod_reduce_mean);
  R("reduce_u8_min",   uint8_t,  lod_dtype_u8,  lod_reduce_min);
  R("reduce_u8_max",   uint8_t,  lod_dtype_u8,  lod_reduce_max);

  R("reduce_i8_mean",  int8_t,   lod_dtype_i8,  lod_reduce_mean);
  R("reduce_i8_min",   int8_t,   lod_dtype_i8,  lod_reduce_min);
  R("reduce_i8_max",   int8_t,   lod_dtype_i8,  lod_reduce_max);

  R("reduce_i16_mean", int16_t,  lod_dtype_i16, lod_reduce_mean);
  R("reduce_i16_min",  int16_t,  lod_dtype_i16, lod_reduce_min);
  R("reduce_i16_max",  int16_t,  lod_dtype_i16, lod_reduce_max);

  R("reduce_u32_mean", uint32_t, lod_dtype_u32, lod_reduce_mean);
  R("reduce_u32_min",  uint32_t, lod_dtype_u32, lod_reduce_min);
  R("reduce_u32_max",  uint32_t, lod_dtype_u32, lod_reduce_max);

  R("reduce_i32_mean", int32_t,  lod_dtype_i32, lod_reduce_mean);
  R("reduce_i32_min",  int32_t,  lod_dtype_i32, lod_reduce_min);
  R("reduce_i32_max",  int32_t,  lod_dtype_i32, lod_reduce_max);

  R("reduce_u64_mean", uint64_t, lod_dtype_u64, lod_reduce_mean);
  R("reduce_u64_min",  uint64_t, lod_dtype_u64, lod_reduce_min);
  R("reduce_u64_max",  uint64_t, lod_dtype_u64, lod_reduce_max);

  R("reduce_i64_mean", int64_t,  lod_dtype_i64, lod_reduce_mean);
  R("reduce_i64_min",  int64_t,  lod_dtype_i64, lod_reduce_min);
  R("reduce_i64_max",  int64_t,  lod_dtype_i64, lod_reduce_max);

  R("reduce_f64_mean", double,   lod_dtype_f64, lod_reduce_mean);
  R("reduce_f64_min",  double,   lod_dtype_f64, lod_reduce_min);
  R("reduce_f64_max",  double,   lod_dtype_f64, lod_reduce_max);
#undef R

  nfail += test_reduce_f16("reduce_f16_mean", lod_reduce_mean);
  nfail += test_reduce_f16("reduce_f16_min",  lod_reduce_min);
  nfail += test_reduce_f16("reduce_f16_max",  lod_reduce_max);

  // --- fold/emit: mean, min, max for each new dtype ---

#define F(label, T, dtype, method)                                             \
  nfail += test_fold<T>(label, dtype, method, 4)

  F("fold_u8_mean",  uint8_t,  lod_dtype_u8,  lod_reduce_mean);
  F("fold_u8_min",   uint8_t,  lod_dtype_u8,  lod_reduce_min);
  F("fold_u8_max",   uint8_t,  lod_dtype_u8,  lod_reduce_max);

  F("fold_i8_mean",  int8_t,   lod_dtype_i8,  lod_reduce_mean);
  F("fold_i8_min",   int8_t,   lod_dtype_i8,  lod_reduce_min);

  F("fold_i16_mean", int16_t,  lod_dtype_i16, lod_reduce_mean);
  F("fold_i16_max",  int16_t,  lod_dtype_i16, lod_reduce_max);

  F("fold_u32_mean", uint32_t, lod_dtype_u32, lod_reduce_mean);
  F("fold_u32_min",  uint32_t, lod_dtype_u32, lod_reduce_min);

  F("fold_i32_mean", int32_t,  lod_dtype_i32, lod_reduce_mean);
  F("fold_i32_max",  int32_t,  lod_dtype_i32, lod_reduce_max);

  F("fold_u64_mean", uint64_t, lod_dtype_u64, lod_reduce_mean);
  F("fold_u64_min",  uint64_t, lod_dtype_u64, lod_reduce_min);

  F("fold_i64_mean", int64_t,  lod_dtype_i64, lod_reduce_mean);

  F("fold_f64_mean", double,   lod_dtype_f64, lod_reduce_mean);
  F("fold_f64_min",  double,   lod_dtype_f64, lod_reduce_min);
  F("fold_f64_max",  double,   lod_dtype_f64, lod_reduce_max);
#undef F

  nfail += test_fold_f16("fold_f16_mean", lod_reduce_mean, 4);
  nfail += test_fold_f16("fold_f16_min",  lod_reduce_min, 4);
  nfail += test_fold_f16("fold_f16_max",  lod_reduce_max, 4);

  log_info("\n%s (%d failures)", nfail ? "FAIL" : "ALL PASSED", nfail);
  cuCtxDestroy(ctx);
  return nfail ? 1 : 0;
}
