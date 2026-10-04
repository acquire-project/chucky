#include "index.ops.util.h"
#include "util/prelude.h"

#ifdef TEST_GPU
#include "gpu/lod.h"
#include "test_runner.h"
#else
#include "cpu/lod.h"
#endif

#include <stdlib.h>
#include <string.h>

// Independent row-major chunk and within-chunk coordinates in storage order.
static uint64_t
expected_offset(int rank,
                const uint64_t* shape,
                const uint64_t* chunks,
                const uint8_t* order,
                uint64_t chunk_stride,
                uint64_t index)
{
  uint64_t coords[HALF_MAX_RANK];
  for (int d = rank - 1; d >= 0; --d) {
    coords[d] = index % shape[d];
    index /= shape[d];
  }
  uint64_t chunk = 0, within = 0;
  for (int j = 0; j < rank; ++j) {
    const int d = order ? order[j] : j;
    chunk = chunk * ceildiv(shape[d], chunks[d]) + coords[d] / chunks[d];
    within = within * chunks[d] + coords[d] % chunks[d];
  }
  return chunk * chunk_stride + within;
}

static int
run_case(uint8_t rank,
         const uint64_t* shape,
         const uint64_t* chunks,
         const uint8_t* order,
         uint8_t bpe)
{
  int error = 1;
  struct tile_stream_layout layout;
  uint64_t* lut = NULL;
  unsigned char *src = NULL, *dst = NULL, *expected = NULL;
#ifdef TEST_GPU
  CUdeviceptr d_src = 0, d_dst = 0, d_lut = 0;
  CUstream stream = 0;
#endif
  CHECK(Fail,
        !test_level_layout(
          &layout, rank, 1, shape, chunks, order, bpe, TEST_CHUNK_ALIGNMENT));
  const uint64_t epoch = layout.epoch_elements;
  const uint64_t width = shape[rank - 1];
  // Start mid-row just before an epoch boundary, then cross several epochs.
  const uint64_t start = epoch > 5 ? epoch - 5 : 1;
  const uint64_t count = 2 * epoch + 17;
  const uint64_t stride = layout.chunks_per_epoch * layout.chunk_stride + 11;
  const uint64_t regions = ceildiv(start + count, epoch);
  const size_t dst_bytes = regions * stride * bpe;
  const size_t lut_bytes = chunk_scatter_lut_bytes(&layout);
  lut = malloc(lut_bytes);
  src = malloc(count * bpe);
  dst = calloc(1, dst_bytes);
  expected = calloc(1, dst_bytes);
  CHECK(Fail, lut && src && dst && expected);
  chunk_scatter_lut_build(&layout, lut);

  for (uint64_t i = 0; i < epoch; ++i)
    CHECK(
      Fail,
      lut[i % width] + lut[width + i / width] ==
        expected_offset(rank, shape, chunks, order, layout.chunk_stride, i));

  for (uint64_t i = 0; i < count * bpe; ++i)
    src[i] = (unsigned char)(1 + i % 251);
  for (uint64_t i = 0; i < count; ++i) {
    const uint64_t index = start + i;
    const uint64_t offset =
      (index / epoch) * stride +
      expected_offset(
        rank, shape, chunks, order, layout.chunk_stride, index % epoch);
    memcpy(expected + offset * bpe, src + i * bpe, bpe);
  }

#ifdef TEST_GPU
  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuMemAlloc(&d_src, count * bpe));
  CU(Fail, cuMemAlloc(&d_dst, dst_bytes));
  CU(Fail, cuMemAlloc(&d_lut, lut_bytes));
  CU(Fail, cuMemcpyHtoDAsync(d_src, src, count * bpe, stream));
  CU(Fail, cuMemcpyHtoDAsync(d_lut, lut, lut_bytes, stream));
  CU(Fail, cuMemsetD8Async(d_dst, 0, dst_bytes, stream));
  CHECK(Fail,
        !scatter_lut_gpu(d_dst,
                         d_src,
                         count,
                         bpe,
                         start,
                         width,
                         epoch,
                         stride * bpe,
                         d_lut,
                         d_lut + width * sizeof(uint64_t),
                         stream));
  CU(Fail, cuStreamSynchronize(stream));
  CU(Fail, cuMemcpyDtoH(dst, d_dst, dst_bytes));
#else
  for (uint64_t i = 0; i < count;) {
    const uint64_t index = start + i;
    uint64_t n = epoch - index % epoch;
    if (n > count - i)
      n = count - i;
    CHECK(Fail,
          !scatter_lut_cpu(dst + (index / epoch) * stride * bpe,
                           src + i * bpe,
                           n,
                           bpe,
                           index % epoch,
                           width,
                           lut,
                           lut + width,
                           NULL));
    i += n;
  }
#endif
  CHECK(Fail, !memcmp(dst, expected, dst_bytes));
#ifndef TEST_GPU
  // Reuse a partially written epoch, clearing only the mapped logical tail.
  CHECK(Fail,
        !scatter_lut_cpu(
          dst, NULL, epoch - start, bpe, start, width, lut, lut + width, NULL));
  for (uint64_t i = start; i < epoch; ++i) {
    const uint64_t offset =
      expected_offset(rank, shape, chunks, order, layout.chunk_stride, i);
    memset(expected + offset * bpe, 0, bpe);
  }
  CHECK(Fail, !memcmp(dst, expected, dst_bytes));
#endif
  error = 0;
Fail:
#ifdef TEST_GPU
  if (stream)
    cuStreamSynchronize(stream);
  cuMemFree(d_src);
  cuMemFree(d_dst);
  cuMemFree(d_lut);
  cuStreamDestroy(stream);
#endif
  free(lut);
  free(src);
  free(dst);
  free(expected);
  return error;
}

static int
test_scatter_lut(void)
{
  for (uint8_t bpe = 1; bpe <= 8; bpe *= 2) {
    CHECK(Fail,
          !run_case(
            3, (uint64_t[]){ 2, 9, 13 }, (uint64_t[]){ 2, 8, 8 }, NULL, bpe));
    CHECK(Fail,
          !run_case(3,
                    (uint64_t[]){ 2, 9, 13 },
                    (uint64_t[]){ 2, 8, 8 },
                    (uint8_t[]){ 0, 2, 1 },
                    bpe));
    CHECK(Fail,
          !run_case(
            3, (uint64_t[]){ 2, 9, 1 }, (uint64_t[]){ 2, 8, 4 }, NULL, bpe));
    CHECK(Fail,
          !run_case(
            3, (uint64_t[]){ 1, 1, 13 }, (uint64_t[]){ 1, 1, 8 }, NULL, bpe));
    CHECK(Fail, !run_case(1, (uint64_t[]){ 7 }, (uint64_t[]){ 7 }, NULL, bpe));
    CHECK(Fail,
          !run_case(10,
                    (uint64_t[]){ 2, 1, 1, 1, 1, 1, 1, 1, 3, 5 },
                    (uint64_t[]){ 2, 1, 1, 1, 1, 1, 1, 1, 2, 4 },
                    (uint8_t[]){ 0, 1, 2, 3, 4, 5, 6, 7, 9, 8 },
                    bpe));
  }
  return 0;
Fail:
  return 1;
}

static int
test_wide_offsets(void)
{
  // Wide chunk/element strides with only eight LUT entries: no huge pool.
  struct tile_stream_layout layout = {
    .lifted_rank = 6,
    .lifted_shape = { 1, 1, 2, 2, 2, 4 },
    .lifted_strides = { 0, 0, 1ll << 34, 1ll << 33, 1ll << 35, 3 },
    .input_shape = { 1, 3, 5 },
    .epoch_elements = 15,
  };
  uint64_t lut[8];
  const uint64_t expected[] = { 0,          3, 6,          9,
                                1ull << 35, 0, 1ull << 33, 1ull << 34 };
  CHECK(Fail, chunk_scatter_lut_bytes(&layout) == sizeof(lut));
  chunk_scatter_lut_build(&layout, lut);
  CHECK(Fail, !memcmp(lut, expected, sizeof(lut)));
  layout.lifted_rank = 2;
  layout.input_shape[0] = layout.epoch_elements = SIZE_MAX / sizeof(uint64_t);
  CHECK(Fail, chunk_scatter_lut_bytes(&layout) == 0);
  return 0;
Fail:
  return 1;
}

#ifdef TEST_GPU
RUN_GPU_TESTS({ "scatter_lut", test_scatter_lut },
              { "wide_offsets", test_wide_offsets })
#else
int
main(void)
{
  return test_scatter_lut() | test_wide_offsets();
}
#endif
