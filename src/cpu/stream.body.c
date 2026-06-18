#include "cpu/stream.body.h"

#include "cpu/transpose.h"
#include "platform/platform.h"
#include "util/metric.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

#include <string.h>

// ---- Pipeline param builders ----

static struct flush_batch_params
make_flush_params(struct cpu_stream_view* v)
{
  const size_t bpe = dtype_bpe(v->config->dtype);
  struct flush_batch_params p = {
    .codec = v->config->codec,
    .bytes_per_element = bpe,
    .chunk_pool = v->chunk_pool,
    .chunk_stride_bytes = v->layout->chunk_stride * bpe,
    .chunk_bytes = v->layout->chunk_elements * bpe,
    .compressed = v->compressed,
    .max_output_size_bytes = v->cl->max_output_size,
    .comp_sizes = v->comp_sizes,
    .total_chunks = v->levels->total_chunks,
    .nlod = v->levels->nlod,
    .cl = v->cl,
    .levels_geo = v->levels,
    .per_lod_agg_layouts = v->agg_layout,
    .h_tail_bytes = v->h_tail_bytes,
    .sink = v->sink,
    .shard_alignment_bytes = v->shard_alignment,
    .pool = v->pool,
    .pool_epochs_scratch = v->pool_epochs_scratch,
    .agg_slots = v->agg_slots,
    .io_done = v->io_done,
    .agg_current = v->agg_current,
    .metrics = v->metrics,
  };
  for (int lv = 0; lv < v->levels->nlod; ++lv) {
    p.levels[lv] = (struct flush_level_view){
      .batch_active_count = v->batch_active_count[lv],
      .chunk_offset = v->levels->level[lv].chunk_offset,
      .shard = &v->shard[lv],
    };
  }
  return p;
}

static struct scatter_epoch_params
make_scatter_params(struct cpu_stream_view* v)
{
  struct scatter_epoch_params p = {
    .dtype = v->config->dtype,
    .reduce_method = v->config->reduce_method,
    .append_reduce_method = v->config->append_reduce_method,
    .cl = v->cl,
    .csrs = v->csrs,
    .chunk_pool = v->chunk_pool,
    .linear = v->linear,
    .lod_values = v->lod_values,
    .scatter_lut = v->scatter_lut,
    .scatter_fixed_dims_offsets = v->scatter_fixed_dims_offsets,
    .append_accum = v->append_accum,
    .append_counts = v->append_counts,
    .pool = v->pool,
    .metrics = v->metrics,
  };
  for (int lv = 0; lv < v->levels->nlod; ++lv) {
    p.morton_lut[lv] = v->morton_lut[lv];
    p.lod_fixed_dims_offsets[lv] = v->lod_fixed_dims_offsets[lv];
  }
  return p;
}

// ---- Debug validation ----

#ifndef NDEBUG
#include <assert.h>
static void
validate_view(const struct cpu_stream_view* v)
{
  assert(v->config && v->sink && v->cl && v->layout && v->levels);
  assert(v->cursor_elements && v->batch_accumulated && v->batch_active_masks);
  assert(v->shard && v->agg_layout && v->batch_active_count && v->io_done);
  assert(v->chunk_pool);
  if (v->levels->enable_multiscale) {
    assert(v->linear && v->lod_values);
    assert(v->scatter_lut && v->scatter_fixed_dims_offsets);
    assert(v->csrs);
  }
}
#else
#define validate_view(v) ((void)0)
#endif

// ---- Shared append body ----

struct writer_result
cpu_stream_append_body(struct cpu_stream_view* v, struct slice input)
{
  validate_view(v);
  const size_t bpe = dtype_bpe(v->config->dtype);
  const uint8_t* src = (const uint8_t*)input.beg;
  const uint8_t* end = (const uint8_t*)input.end;
  const uint64_t total_limit = v->total_element_limit;

  while (src < end) {
    // Capacity reached: refuse further writes and report `finished` with the
    // remaining input unconsumed. Sink finalization is NOT run here — it
    // happens on explicit `writer_flush` or on stream destroy. Keeping the
    // producer path free of sink finalization means appends never block on
    // IO the stream's owner hasn't asked for.
    if (total_limit > 0 && *v->cursor_elements >= total_limit)
      return writer_finished_at(src, end);

    const uint64_t epoch_remaining =
      v->layout->epoch_elements -
      (*v->cursor_elements % v->layout->epoch_elements);
    const uint64_t input_remaining = (uint64_t)(end - src) / bpe;
    uint64_t elements =
      epoch_remaining < input_remaining ? epoch_remaining : input_remaining;

    if (total_limit > 0) {
      uint64_t cap = total_limit - *v->cursor_elements;
      if (elements > cap)
        elements = cap;
    }

    const uint64_t bytes = elements * bpe;

    // Scatter into chunk pool (or LOD buffer for multiscale).
    {
      struct platform_clock clk = { 0 };
      platform_toc(&clk);

      if (v->levels->enable_multiscale) {
        uint64_t epoch_offset = *v->cursor_elements % v->layout->epoch_elements;
        memcpy((char*)v->linear + epoch_offset * bpe, src, bytes);
      } else {
        void* epoch_pool =
          (char*)v->chunk_pool + (uint64_t)*v->batch_accumulated *
                                   v->levels->total_chunks *
                                   v->layout->chunk_stride * bpe;
        CHECK(Error,
              transpose_cpu(epoch_pool,
                            src,
                            bytes,
                            (uint8_t)bpe,
                            *v->cursor_elements,
                            v->layout->lifted_rank,
                            v->layout->lifted_shape,
                            v->layout->lifted_strides,
                            v->pool) == 0);
      }

      float ms = (float)(platform_toc(&clk) * 1000.0);
      if (v->metrics)
        accumulate_metric_ms(&v->metrics->scatter, ms, bytes, 0);
    }

    *v->cursor_elements += elements;
    src += bytes;

    // Epoch boundary: accumulate into batch, flush when full.
    if (*v->cursor_elements % v->layout->epoch_elements == 0 &&
        *v->cursor_elements > 0) {
      uint32_t active_mask = 1;
      if (v->levels->enable_multiscale) {
        struct scatter_epoch_params sp = make_scatter_params(v);
        CHECK(Error,
              cpu_pipeline_scatter_epoch(
                &sp, *v->batch_accumulated, &active_mask) == 0);
      }

      CHECK(Error, *v->batch_accumulated < v->cl->epochs_per_batch);
      v->batch_active_masks[*v->batch_accumulated] = active_mask;
      (*v->batch_accumulated)++;

      if (*v->batch_accumulated == v->cl->epochs_per_batch) {
        struct flush_batch_params fp = make_flush_params(v);
        CHECK(Error,
              cpu_pipeline_flush_batch(
                &fp, *v->batch_accumulated, v->batch_active_masks) == 0);
        *v->batch_accumulated = 0;

        if (!v->pool_fully_covered)
          memset(v->chunk_pool,
                 0,
                 (uint64_t)v->cl->epochs_per_batch * v->levels->total_chunks *
                   v->layout->chunk_stride * bpe);
      }

      if (v->lod_values) {
        size_t lod_bytes =
          v->cl->plan.level_spans.ends[v->cl->plan.levels.nlod - 1] * bpe;
        memset(v->lod_values, 0, lod_bytes);
      }

      // Periodic metadata update.
      if (v->metadata_update_clock && v->sink->update_append) {
        struct platform_clock peek = *v->metadata_update_clock;
        float elapsed = platform_toc(&peek);
        if (elapsed >= v->config->metadata_update_interval_s) {
          *v->metadata_update_clock = peek;
          const uint8_t na = dim_info_n_append(&v->cl->dims);
          for (int lv = 0; lv < v->levels->nlod; ++lv) {
            struct shard_state* ss = &v->shard[lv];
            uint64_t total_ac = ss->shard_epoch * ss->chunks_per_shard_append +
                                ss->epoch_in_shard;
            uint64_t append_sizes[HALF_MAX_RANK];
            dim_info_decompose_append_sizes(
              &v->cl->dims, total_ac, append_sizes);
            if (v->sink->update_append(v->sink, (uint8_t)lv, na, append_sizes))
              goto Error;
          }
        }
      }
    }
  }

  return (struct writer_result){ .error = 0,
                                 .rest = { .beg = src, .end = end } };

Error:
  return writer_error_at(src, end);
}

// ---- Batch-only flush (for multiarray switch) ----

int
cpu_stream_flush_batch(struct cpu_stream_view* v)
{
  if (*v->batch_accumulated == 0)
    return 0;
  struct flush_batch_params fp = make_flush_params(v);
  if (cpu_pipeline_flush_batch(
        &fp, *v->batch_accumulated, v->batch_active_masks))
    return 1;
  *v->batch_accumulated = 0;
  return 0;
}

// ---- Shared flush body ----

struct writer_result
cpu_stream_flush_body(struct cpu_stream_view* v)
{
  // Flush partial epoch into the batch.
  if (*v->cursor_elements % v->layout->epoch_elements != 0) {
    uint32_t active_mask = 1;
    if (v->levels->enable_multiscale) {
      struct scatter_epoch_params sp = make_scatter_params(v);
      if (cpu_pipeline_scatter_epoch(&sp, *v->batch_accumulated, &active_mask))
        return writer_error();
    }
    if (*v->batch_accumulated >= v->cl->epochs_per_batch)
      return writer_error();
    v->batch_active_masks[*v->batch_accumulated] = active_mask;
    (*v->batch_accumulated)++;
  }

  // Flush any accumulated batch.
  if (*v->batch_accumulated > 0) {
    struct flush_batch_params fp = make_flush_params(v);
    if (cpu_pipeline_flush_batch(
          &fp, *v->batch_accumulated, v->batch_active_masks))
      return writer_error();
    *v->batch_accumulated = 0;
  }

  // Drain any partial append accumulators.
  if (v->cl->dims.append_downsample && v->append_accum) {
    struct append_drain_params dp = {
      .cl = v->cl,
      .dtype = v->config->dtype,
      .append_reduce_method = v->config->append_reduce_method,
      .lod_values = v->lod_values,
      .append_accum = v->append_accum,
      .append_counts = v->append_counts,
      .chunk_pool = v->chunk_pool,
      .pool = v->pool,
      .metrics = v->metrics,
    };
    for (int lv = 0; lv < v->levels->nlod; ++lv) {
      dp.morton_lut[lv] = v->morton_lut[lv];
      dp.lod_fixed_dims_offsets[lv] = v->lod_fixed_dims_offsets[lv];
    }

    uint32_t drain_mask = 0;
    if (cpu_pipeline_append_drain(&dp, &drain_mask))
      return writer_error();

    if (drain_mask) {
      v->batch_active_masks[0] = drain_mask;
      struct flush_batch_params fp = make_flush_params(v);
      if (cpu_pipeline_flush_batch(&fp, 1, v->batch_active_masks))
        return writer_error();
    }
  }

  // Emit partial shards.
  {
    struct platform_clock emit_clk = { 0 };
    platform_toc(&emit_clk);

    // Both slots may hold in-flight IO from this batch; wait on both before
    // finalize. Single fence per slot covers all LODs (one shared IO queue).
    if (v->sink->wait_fence) {
      v->sink->wait_fence(v->sink, v->io_done[0]);
      v->sink->wait_fence(v->sink, v->io_done[1]);
    }

    if (v->sink->has_error && v->sink->has_error(v->sink))
      return writer_error();

    for (int lv = 0; lv < v->levels->nlod; ++lv) {
      if (v->shard[lv].epoch_in_shard > 0) {
        if (finalize_shards(&v->shard[lv], v->sink, v->shard_alignment)) {
          // finalize_shards can queue footer write_direct jobs that reference
          // footer_buf_pool, then fail. Drain before returning so destroy can
          // free that pool with no IO still pointing into it.
          shard_sink_drain(v->sink);
          return writer_error();
        }
      }
    }

    if (shard_sink_drain(v->sink))
      return writer_error();

    float emit_ms = (float)(platform_toc(&emit_clk) * 1000.0);
    if (v->metrics)
      accumulate_metric_ms(&v->metrics->sink, emit_ms, 0, 0);
  }

  // Final metadata.
  if (v->sink->update_append) {
    const uint8_t na = dim_info_n_append(&v->cl->dims);
    for (int lv = 0; lv < v->levels->nlod; ++lv) {
      uint64_t append_sizes[HALF_MAX_RANK];
      dim_info_final_append_sizes(
        &v->cl->dims, *v->cursor_elements, lv, append_sizes);
      if (v->sink->update_append(v->sink, (uint8_t)lv, na, append_sizes))
        return writer_error();
    }
  }

  if (v->sink->flush && v->sink->flush(v->sink))
    return writer_error();

  return writer_ok();
}
