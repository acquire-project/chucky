#include "gpu/stream.lod.h"

#include "dimension.h"
#include "gpu/lod.h"
#include "lod/lod_plan.h"
#include "gpu/prelude.cuda.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

// --- LOD init: GPU uploads and LUT building from pre-computed plan/layouts ---

// Upload plan shapes to GPU. Plan must already be initialized.
static int
upload_plan_shapes(struct lod_state* lod, uint8_t rank)
{
  CU(Fail, cuMemAlloc(&lod->d_full_shape, rank * sizeof(uint64_t)));
  CU(Fail,
     cuMemcpyHtoD(
       lod->d_full_shape, lod->plan.shapes[0], rank * sizeof(uint64_t)));

  if (lod->plan.lod_ndim > 0) {
    CU(Fail,
       cuMemAlloc(&lod->d_lod_shape, lod->plan.lod_ndim * sizeof(uint64_t)));
    CU(Fail,
       cuMemcpyHtoD(lod->d_lod_shape,
                    lod->plan.lod_shapes[0],
                    lod->plan.lod_ndim * sizeof(uint64_t)));
  }

  return 0;
Fail:
  return 1;
}

// Compute LOD strides, upload to GPU, and build gather LUT.
// Owns d_lod_strides — freed in both success and failure paths.
static int
build_gather_lut_with_strides(struct lod_state* lod,
                              const struct lod_plan* p,
                              const uint64_t* shape,
                              uint8_t rank)
{
  CUdeviceptr d_lod_strides = 0;

  uint64_t full_strides[LOD_MAX_NDIM];
  full_strides[rank - 1] = 1;
  for (int d = rank - 2; d >= 0; --d)
    full_strides[d] = full_strides[d + 1] * shape[d + 1];

  uint64_t lod_strides[LOD_MAX_NDIM];
  int li = p->lod_ndim - 1;
  for (int d = rank - 1; d >= 0; --d) {
    if ((p->lod_mask >> d) & 1) {
      lod_strides[li] = full_strides[d];
      li--;
    }
  }

  CU(Fail, cuMemAlloc(&d_lod_strides, p->lod_ndim * sizeof(uint64_t)));
  CU(Fail,
     cuMemcpyHtoD(d_lod_strides, lod_strides, p->lod_ndim * sizeof(uint64_t)));

  CU(Fail, cuMemAlloc(&lod->d_gather_lut, p->lod_nelem[0] * sizeof(uint32_t)));
  CHECK(Fail,
        lod_build_gather_lut(lod->d_gather_lut,
                             lod->d_lod_shape,
                             d_lod_strides,
                             p->lod_ndim,
                             p->lod_shapes[0],
                             p->lod_nelem[0],
                             0) == 0);

  cuMemFree(d_lod_strides);
  return 0;
Fail:
  cuMemFree(d_lod_strides);
  return 1;
}

static int
init_gather_lut(struct lod_state* lod,
                const struct tile_stream_configuration* config)
{
  if (lod->plan.lod_ndim == 0)
    return 0;

  const uint8_t rank = config->rank;
  const struct dimension* dims = config->dimensions;
  struct lod_plan* p = &lod->plan;

  uint64_t shape[HALF_MAX_RANK];
  const uint8_t na = dims_n_append(dims, rank);
  for (int d = 0; d < na; ++d)
    shape[d] = dims[d].chunk_size;
  for (int d = na; d < rank; ++d)
    shape[d] = dims[d].size;

  {
    uint64_t epoch_elements = 1;
    for (int d = 0; d < rank; ++d)
      epoch_elements *= shape[d];
    if (epoch_elements > UINT32_MAX) {
      log_error("epoch_elements %llu exceeds uint32_t limit for gather LUT",
                (unsigned long long)epoch_elements);
      goto Fail;
    }
  }

  CHECK(Fail, build_gather_lut_with_strides(lod, p, shape, rank) == 0);

  // Compute batch_offsets on host and upload
  {
    uint32_t* batch_offsets =
      (uint32_t*)calloc(p->batch_count, sizeof(uint32_t));
    CHECK(Fail, batch_offsets);

    uint64_t full_strides[LOD_MAX_NDIM];
    full_strides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; --d)
      full_strides[d] = full_strides[d + 1] * shape[d + 1];

    for (uint64_t bi = 0; bi < p->batch_count; ++bi) {
      uint64_t remainder = bi;
      uint64_t offset = 0;
      for (int k = p->batch_ndim - 1; k >= 0; --k) {
        uint64_t coord = remainder % p->batch_shape[k];
        remainder /= p->batch_shape[k];
        offset += coord * full_strides[p->batch_map[k]];
      }
      batch_offsets[bi] = (uint32_t)offset;
    }

    CU(Fail,
       cuMemAlloc(&lod->d_batch_offsets, p->batch_count * sizeof(uint32_t)));
    CU(Fail,
       cuMemcpyHtoD(lod->d_batch_offsets,
                    batch_offsets,
                    p->batch_count * sizeof(uint32_t)));
    free(batch_offsets);
  }

  return 0;
Fail:
  return 1;
}

static int
init_reduce_level_arrays(struct lod_state* lod)
{
  for (int l = 0; l < lod->plan.nlod - 1; ++l) {
    struct lod_span seg = lod_segment(&lod->plan, l);
    uint64_t n_parents = lod_span_len(seg);

    CU(Fail,
       cuMemAlloc(&lod->d_child_shapes[l],
                  lod->plan.lod_ndim * sizeof(uint64_t)));
    CU(Fail,
       cuMemcpyHtoD(lod->d_child_shapes[l],
                    lod->plan.lod_shapes[l],
                    lod->plan.lod_ndim * sizeof(uint64_t)));

    CU(Fail,
       cuMemAlloc(&lod->d_parent_shapes[l],
                  lod->plan.lod_ndim * sizeof(uint64_t)));
    CU(Fail,
       cuMemcpyHtoD(lod->d_parent_shapes[l],
                    lod->plan.lod_shapes[l + 1],
                    lod->plan.lod_ndim * sizeof(uint64_t)));

    CU(Fail, cuMemAlloc(&lod->d_level_ends[l], n_parents * sizeof(uint64_t)));
  }

  return 0;
Fail:
  return 1;
}

// Upload pre-computed LOD level layouts to GPU.
// Host fields in lod->layouts must already be populated.
static int
upload_lod_level_layouts(struct lod_state* lod)
{
  for (int lv = 0; lv < lod->plan.nlod; ++lv) {
    const struct tile_stream_layout* layout = &lod->layouts[lv];
    struct tile_stream_layout_gpu* gpu = &lod->layout_gpu[lv];
    const size_t sb = layout->lifted_rank * sizeof(uint64_t);
    const size_t stb = layout->lifted_rank * sizeof(int64_t);
    CU(Fail, cuMemAlloc((CUdeviceptr*)&gpu->d_lifted_shape, sb));
    CU(Fail, cuMemAlloc((CUdeviceptr*)&gpu->d_lifted_strides, stb));
    CU(Fail,
       cuMemcpyHtoD(
         (CUdeviceptr)gpu->d_lifted_shape, layout->lifted_shape, sb));
    CU(Fail,
       cuMemcpyHtoD(
         (CUdeviceptr)gpu->d_lifted_strides, layout->lifted_strides, stb));
  }

  return 0;
Fail:
  return 1;
}

// Upload chunk sizes/strides to GPU and build chunk scatter LUT.
// Owns d_chunk_sizes and d_chunk_strides — freed in both success and failure.
static int
build_chunk_scatter_with_temps(struct lod_state* lod,
                               const struct lod_plan* p,
                               const struct tile_stream_layout* lay,
                               CUdeviceptr d_lod_shape_lv,
                               int lv)
{
  CUdeviceptr d_chunk_sizes = 0, d_chunk_strides = 0;
  uint64_t lod_count = p->lod_nelem[lv];

  uint64_t lod_chunk_sizes[LOD_MAX_NDIM];
  int64_t lod_chunk_strides[2 * LOD_MAX_NDIM];

  for (int li = 0; li < p->lod_ndim; ++li) {
    int d = p->lod_map[li];
    lod_chunk_sizes[li] = lay->lifted_shape[2 * d + 1];
    lod_chunk_strides[2 * li] = lay->lifted_strides[2 * d];
    lod_chunk_strides[2 * li + 1] = lay->lifted_strides[2 * d + 1];
  }

  CU(Fail, cuMemAlloc(&d_chunk_sizes, p->lod_ndim * sizeof(uint64_t)));
  CU(Fail, cuMemAlloc(&d_chunk_strides, 2 * p->lod_ndim * sizeof(int64_t)));
  CU(Fail,
     cuMemcpyHtoD(
       d_chunk_sizes, lod_chunk_sizes, p->lod_ndim * sizeof(uint64_t)));
  CU(Fail,
     cuMemcpyHtoD(
       d_chunk_strides, lod_chunk_strides, 2 * p->lod_ndim * sizeof(int64_t)));

  CU(Fail,
     cuMemAlloc(&lod->d_morton_chunk_lut[lv], lod_count * sizeof(uint32_t)));
  CHECK(Fail,
        lod_build_chunk_scatter_lut(lod->d_morton_chunk_lut[lv],
                                    d_lod_shape_lv,
                                    d_chunk_sizes,
                                    d_chunk_strides,
                                    p->lod_ndim,
                                    p->lod_shapes[lv],
                                    lod_count,
                                    0) == 0);

  cuMemFree(d_chunk_sizes);
  cuMemFree(d_chunk_strides);
  return 0;
Fail:
  cuMemFree(d_chunk_sizes);
  cuMemFree(d_chunk_strides);
  return 1;
}

// Build morton-to-chunk LUT for a single level. Returns 0 on success.
static int
build_morton_lut_for_level(struct lod_state* lod,
                           const struct tile_stream_layout* lay,
                           int lv)
{
  struct lod_plan* p = &lod->plan;

  // Upload LOD shape to device (temporary for LUT building)
  CUdeviceptr d_lod_shape_lv = 0;
  int free_shape = 0;
  if (lv == 0) {
    d_lod_shape_lv = lod->d_lod_shape;
  } else {
    const size_t lod_shape_bytes = p->lod_ndim * sizeof(uint64_t);
    CU(Fail, cuMemAlloc(&d_lod_shape_lv, lod_shape_bytes));
    free_shape = 1;
    CU(FailShape,
       cuMemcpyHtoD(d_lod_shape_lv, p->lod_shapes[lv], lod_shape_bytes));
  }

  CHECK(FailShape,
        build_chunk_scatter_with_temps(lod, p, lay, d_lod_shape_lv, lv) == 0);

  if (free_shape)
    cuMemFree(d_lod_shape_lv);

  // Compute batch_chunk_offsets on host and upload
  {
    uint32_t* batch_offsets =
      (uint32_t*)calloc(p->batch_count, sizeof(uint32_t));
    CHECK(Fail, batch_offsets);

    for (uint64_t bi = 0; bi < p->batch_count; ++bi) {
      uint64_t remainder = bi;
      int64_t offset = 0;
      for (int k = p->batch_ndim - 1; k >= 0; --k) {
        uint64_t coord = remainder % p->batch_shape[k];
        remainder /= p->batch_shape[k];
        int d = p->batch_map[k];
        uint64_t chunk_idx = coord / lay->lifted_shape[2 * d + 1];
        uint64_t within = coord % lay->lifted_shape[2 * d + 1];
        offset += (int64_t)chunk_idx * lay->lifted_strides[2 * d];
        offset += (int64_t)within * lay->lifted_strides[2 * d + 1];
      }
      batch_offsets[bi] = (uint32_t)offset;
    }

    CU(Fail,
       cuMemAlloc(&lod->d_morton_batch_chunk_offsets[lv],
                  p->batch_count * sizeof(uint32_t)));
    CU(Fail,
       cuMemcpyHtoD(lod->d_morton_batch_chunk_offsets[lv],
                    batch_offsets,
                    p->batch_count * sizeof(uint32_t)));
    free(batch_offsets);
  }

  return 0;
FailShape:
  if (free_shape)
    cuMemFree(d_lod_shape_lv);
Fail:
  return 1;
}

static int
init_morton_scatter_luts(struct lod_state* lod)
{
  if (lod->plan.lod_ndim == 0)
    return 0;

  for (int lv = 0; lv < lod->plan.nlod; ++lv) {
    CHECK(Fail, build_morton_lut_for_level(lod, &lod->layouts[lv], lv) == 0);
  }

  return 0;
Fail:
  return 1;
}

int
lod_state_init(struct lod_state* lod,
               struct level_geometry* levels,
               const struct tile_stream_configuration* config)
{
  // Always upload level layouts (L0 included).
  CHECK(Fail, upload_lod_level_layouts(lod) == 0);

  if (!levels->enable_multiscale)
    return 0;

  // LOD-specific: plan shapes, gather LUT, reduce arrays, morton LUTs.
  for (int k = 0; k < lod->plan.nlod; ++k) {
    if (lod->plan.lod_nelem[k] > UINT32_MAX) {
      log_error("LOD level %d count %llu exceeds uint32_t limit",
                k,
                (unsigned long long)lod->plan.lod_nelem[k]);
      goto Fail;
    }
  }

  CHECK(Fail, upload_plan_shapes(lod, config->rank) == 0);
  CHECK(Fail, init_gather_lut(lod, config) == 0);
  CHECK(Fail, init_reduce_level_arrays(lod) == 0);
  CHECK(Fail, init_morton_scatter_luts(lod) == 0);

  levels->nlod = lod->plan.nlod;
  return 0;
Fail:
  return 1;
}

// --- LOD buffer allocation ---

int
lod_state_init_buffers(struct lod_state* lod,
                       enum dtype dtype)
{
  const size_t bytes_per_element = dtype_bpe(dtype);
  size_t linear_bytes = lod->layouts[0].epoch_elements * bytes_per_element;
  CU(Fail, cuMemAlloc(&lod->d_linear, linear_bytes));

  uint64_t total_vals = lod->plan.levels.ends[lod->plan.nlod - 1];
  size_t morton_bytes = total_vals * bytes_per_element;
  CU(Fail, cuMemAlloc(&lod->d_morton, morton_bytes));

  CU(Fail, cuEventCreate(&lod->t_start, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&lod->t_scatter_end, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&lod->t_reduce_end, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&lod->t_append_end, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&lod->t_end, CU_EVENT_DEFAULT));

  return 0;
Fail:
  return 1;
}

// --- Append-dim accumulator allocation ---

int
lod_state_init_accumulators(struct lod_state* lod,
                            const struct tile_stream_configuration* config)
{
  struct lod_plan* p = &lod->plan;

  lod->append_accum.morton_offset = p->levels.ends[0];

  lod->append_accum.total_elements = 0;
  for (int lv = 1; lv < p->nlod; ++lv)
    lod->append_accum.total_elements += p->batch_count * p->lod_nelem[lv];

  if (lod->append_accum.total_elements == 0)
    return 0;

  size_t accum_bpe = dtype_accum_bpe(config->dtype, config->append_reduce_method);
  size_t accum_bytes = lod->append_accum.total_elements * accum_bpe;
  CU(Fail, cuMemAlloc(&lod->append_accum.d_accum, accum_bytes));

  {
    uint8_t* h_level_ids = (uint8_t*)malloc(lod->append_accum.total_elements);
    CHECK(Fail, h_level_ids);

    uint64_t offset = 0;
    for (int lv = 1; lv < p->nlod; ++lv) {
      uint64_t n = p->batch_count * p->lod_nelem[lv];
      memset(h_level_ids + offset, (uint8_t)lv, n);
      offset += n;
    }

    CUresult r = cuMemAlloc(&lod->append_accum.d_level_ids, lod->append_accum.total_elements);
    if (r != CUDA_SUCCESS) {
      free(h_level_ids);
      goto Fail;
    }
    r = cuMemcpyHtoD(
      lod->append_accum.d_level_ids, h_level_ids, lod->append_accum.total_elements);
    free(h_level_ids);
    if (r != CUDA_SUCCESS)
      goto Fail;
  }

  {
    CU(Fail,
       cuMemAlloc(&lod->append_accum.d_counts, (uint64_t)p->nlod * sizeof(uint32_t)));
    memset(lod->append_accum.counts, 0, sizeof(lod->append_accum.counts));
    CU(Fail,
       cuMemcpyHtoD(lod->append_accum.d_counts,
                    lod->append_accum.counts,
                    (uint64_t)p->nlod * sizeof(uint32_t)));
  }

  return 0;
Fail:
  return 1;
}

// --- LOD destroy ---

void
lod_state_destroy(struct lod_state* lod)
{
  // Append-dim accumulation state
  if (lod->append_accum.d_accum)
    CUWARN(cuMemFree(lod->append_accum.d_accum));
  if (lod->append_accum.d_level_ids)
    CUWARN(cuMemFree(lod->append_accum.d_level_ids));
  if (lod->append_accum.d_counts)
    CUWARN(cuMemFree(lod->append_accum.d_counts));

  // LOD cleanup
  if (lod->d_linear)
    CUWARN(cuMemFree(lod->d_linear));
  if (lod->d_morton)
    CUWARN(cuMemFree(lod->d_morton));
  CUWARN(cuMemFree(lod->d_full_shape));
  CUWARN(cuMemFree(lod->d_lod_shape));
  CUWARN(cuMemFree(lod->d_gather_lut));
  CUWARN(cuMemFree(lod->d_batch_offsets));
  for (int i = 0; i < lod->plan.nlod; ++i) {
    CUWARN(cuMemFree(lod->d_morton_chunk_lut[i]));
    CUWARN(cuMemFree(lod->d_morton_batch_chunk_offsets[i]));
  }
  for (int i = 0; i < lod->plan.nlod - 1; ++i) {
    CUWARN(cuMemFree(lod->d_child_shapes[i]));
    CUWARN(cuMemFree(lod->d_parent_shapes[i]));
    CUWARN(cuMemFree(lod->d_level_ends[i]));
  }
  for (int i = 0; i < lod->plan.nlod; ++i) {
    CUWARN(cuMemFree((CUdeviceptr)lod->layout_gpu[i].d_lifted_shape));
    CUWARN(cuMemFree((CUdeviceptr)lod->layout_gpu[i].d_lifted_strides));
  }
  if (lod->t_start) {
    CUWARN(cuEventDestroy(lod->t_start));
    CUWARN(cuEventDestroy(lod->t_scatter_end));
    CUWARN(cuEventDestroy(lod->t_reduce_end));
    CUWARN(cuEventDestroy(lod->t_append_end));
    CUWARN(cuEventDestroy(lod->t_end));
  }
  lod_plan_free(&lod->plan);
}

// --- LOD runtime ---

// Append-dim fold + emit for append-dim downsampling.
//
// Each LOD level l>0 accumulates 2^l inner-reduced epochs before emitting.
// A running accumulator (wider type for mean, native for min/max) is
// maintained per level. On each epoch: (1) fold new inner-reduced data into
// the accumulator via lod_accum_fold_fused, (2) for any level whose
// count reaches its period, emit the finalized result back into the
// morton buffer and reset the counter.
//
// *out_mask is OR'd with (1u << lv) for each level that emitted.
static int
run_append_fold_emit(struct lod_state* lod,
                   enum dtype dtype,
                   enum lod_reduce_method append_reduce_method,
                   CUstream compute,
                   uint32_t* out_mask)
{
  struct lod_plan* p = &lod->plan;
  const size_t bytes_per_element = dtype_bpe(dtype);

  // Upload current counts to device before fused kernel
  CU(Error,
     cuMemcpyHtoDAsync(lod->append_accum.d_counts,
                       lod->append_accum.counts,
                       p->nlod * sizeof(uint32_t),
                       compute));

  // Single fused fold over all levels 1+
  CUdeviceptr morton_1plus = lod->d_morton + lod->append_accum.morton_offset * bytes_per_element;
  CHECK(Error,
        lod_accum_fold_fused(lod->append_accum.d_accum,
                             morton_1plus,
                             lod->append_accum.d_level_ids,
                             lod->append_accum.d_counts,
                             dtype,
                             append_reduce_method,
                             lod->append_accum.total_elements,
                             compute) == 0);

  // Increment counts, emit ready levels back to morton
  for (int lv = 1; lv < p->nlod; ++lv) {
    lod->append_accum.counts[lv]++;
    uint32_t period = 1u << lv;

    if (lod->append_accum.counts[lv] >= period) {
      struct lod_span lev = lod_spans_at(&p->levels, lv);
      uint64_t n_elements = p->batch_count * p->lod_nelem[lv];

      uint64_t accum_offset = 0;
      for (int k = 1; k < lv; ++k)
        accum_offset += p->batch_count * p->lod_nelem[k];

      size_t accum_bpe = dtype_accum_bpe(dtype, append_reduce_method);

      CUdeviceptr morton_lv = lod->d_morton + lev.beg * bytes_per_element;
      CUdeviceptr accum_lv = lod->append_accum.d_accum + accum_offset * accum_bpe;

      CHECK(Error,
            lod_accum_emit(morton_lv,
                           accum_lv,
                           dtype,
                           append_reduce_method,
                           n_elements,
                           lod->append_accum.counts[lv],
                           compute) == 0);

      lod->append_accum.counts[lv] = 0;
      *out_mask |= (1u << lv);
    }
  }

  return 0;

Error:
  return 1;
}

// Scatter morton-ordered LOD data into the chunk pool for all active levels.
static int
scatter_morton_to_chunks(struct lod_state* lod,
                         const struct level_geometry* levels,
                         void* pool_epoch,
                         enum dtype dtype,
                         uint32_t active_levels_mask,
                         CUstream compute)
{
  struct lod_plan* p = &lod->plan;
  const size_t bytes_per_element = dtype_bpe(dtype);
  const uint64_t chunk_stride = lod->layouts[0].chunk_stride;

  for (int lv = 0; lv < p->nlod; ++lv) {
    if (!(active_levels_mask & (1u << lv)))
      continue;

    struct lod_span lev = lod_spans_at(&p->levels, lv);
    CUdeviceptr dst = (CUdeviceptr)pool_epoch +
                      levels->chunk_offset[lv] * chunk_stride * bytes_per_element;

    CHECK(Error,
          lod_morton_to_chunks_lut(dst,
                                   lod->d_morton + lev.beg * bytes_per_element,
                                   lod->d_morton_chunk_lut[lv],
                                   lod->d_morton_batch_chunk_offsets[lv],
                                   dtype,
                                   p->lod_nelem[lv],
                                   p->batch_count,
                                   compute) == 0);
  }

  return 0;

Error:
  return 1;
}

int
lod_run_epoch(struct lod_state* lod,
              const struct level_geometry* levels,
              void* pool_epoch,
              enum dtype dtype,
              enum lod_reduce_method reduce_method,
              enum lod_reduce_method append_reduce_method,
              CUstream compute,
              uint32_t* out_active_mask)
{
  struct lod_plan* p = &lod->plan;

  CU(Error, cuEventRecord(lod->t_start, compute));

  CHECK(Error,
        lod_gather_lut(lod->d_morton,
                       lod->d_linear,
                       lod->d_gather_lut,
                       lod->d_batch_offsets,
                       dtype,
                       p->lod_nelem[0],
                       p->batch_count,
                       compute) == 0);

  CU(Error, cuEventRecord(lod->t_scatter_end, compute));

  for (int l = 0; l < p->nlod - 1; ++l) {
    struct lod_span seg = lod_segment(p, l);
    uint64_t n_parents = lod_span_len(seg);

    CHECK(Error,
          lod_fill_ends_gpu(lod->d_level_ends[l],
                            p->lod_ndim,
                            lod->d_child_shapes[l],
                            lod->d_parent_shapes[l],
                            p->lod_shapes[l],
                            p->lod_shapes[l + 1],
                            n_parents,
                            compute) == 0);

    struct lod_span src_level = lod_spans_at(&p->levels, l);
    struct lod_span dst_level = lod_spans_at(&p->levels, l + 1);

    CHECK(Error,
          lod_reduce(lod->d_morton,
                     lod->d_level_ends[l],
                     dtype,
                     reduce_method,
                     src_level.beg,
                     dst_level.beg,
                     p->lod_nelem[l],
                     p->lod_nelem[l + 1],
                     p->batch_count,
                     compute) == 0);
  }

  CU(Error, cuEventRecord(lod->t_reduce_end, compute));

  uint32_t active_levels_mask = 1; // L0 always active
  if (levels->append_downsample && lod->append_accum.total_elements > 0) {
    CHECK(Error,
          run_append_fold_emit(
            lod, dtype, append_reduce_method, compute, &active_levels_mask) == 0);
  }

  CU(Error, cuEventRecord(lod->t_append_end, compute));

  CHECK(
    Error,
    scatter_morton_to_chunks(
      lod, levels, pool_epoch, dtype, active_levels_mask, compute) == 0);

  CU(Error, cuEventRecord(lod->t_end, compute));

  *out_active_mask = active_levels_mask;
  return 0;

Error:
  return 1;
}
