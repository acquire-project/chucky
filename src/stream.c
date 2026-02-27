#include "compress.h"
#include "lod.h"
#include "metric.cuda.h"
#include "platform.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"

#include <stdlib.h>
#include <string.h>

static struct writer_result
writer_ok(void)
{
  return (struct writer_result){ 0 };
}

static struct writer_result
writer_error(void)
{
  return (struct writer_result){ .error = 1 };
}

static struct writer_result
writer_error_at(const void* beg, const void* end)
{
  return (struct writer_result){ .error = 1, .rest = { beg, end } };
}

static void
buffer_free(struct buffer* buffer)
{
  if (!buffer || !buffer->data) {
    return;
  }

  if (buffer->ready) {
    CUresult res = cuEventDestroy(buffer->ready);
    if (res != CUDA_SUCCESS) {
      const char* err_str = NULL;
      cuGetErrorString(res, &err_str);
      log_warn("Failed to destroy event: %s", err_str ? err_str : "unknown");
    }
    buffer->ready = NULL;
  }

  switch (buffer->domain) {
    case host:
      cuMemFreeHost(buffer->data);
      break;
    case device:
      cuMemFree((CUdeviceptr)buffer->data);
      break;
    default:
      log_error("Invalid domain during buffer_free: %d", buffer->domain);
      return;
  }

  buffer->data = NULL;
}

static struct buffer
buffer_new(size_t capacity, enum domain domain, unsigned int host_flags)
{
  struct buffer buf = { 0 };
  buf.domain = domain;

  switch (domain) {
    case host:
      CU(Fail, cuMemHostAlloc(&buf.data, capacity, host_flags));
      break;
    case device:
      CU(Fail, cuMemAlloc((CUdeviceptr*)&buf.data, capacity));
      break;
    default:
      log_error("Invalid domain: %d", domain);
      goto Fail;
  }
  CU(Fail, cuEventCreate(&buf.ready, CU_EVENT_DEFAULT));
  return buf;

Fail:
  buffer_free(&buf);
  return (struct buffer){ 0 };
}

// FIXME: don't need this function
static void
device_free(void* ptr)
{
  CUWARN(cuMemFree((CUdeviceptr)ptr));
}

// --- Helpers ---

// Return pointer to the current L0 tile pool (A if pool_current==0, B if 1).
static inline void*
current_pool(struct transpose_stream* s)
{
  return s->pool_current ? s->pool_B.data : s->pool_A.data;
}


// Dispatch staged data: H2D transfer + scatter kernel
// Returns 0 on success, 1 on error.
static int
dispatch_scatter(struct transpose_stream* s)
{
  const size_t bpe = s->config.bytes_per_element;
  if (bpe == 0)
    return 0;

  const uint64_t elements = s->stage.fill / bpe;
  if (elements == 0)
    return 0;

  const int idx = s->stage.current;
  struct staging_slot* ss = &s->stage.slot[idx];

  void* pool = current_pool(s);

  // H2D — wait for prior scatter to finish reading d_in before overwriting
  CU(Error, cuStreamWaitEvent(s->h2d, ss->t_scatter_end, 0));
  CU(Error, cuEventRecord(ss->t_h2d_start, s->h2d));
  CU(Error,
     cuMemcpyHtoDAsync(
       (CUdeviceptr)ss->d_in.data, ss->h_in.data, s->stage.fill, s->h2d));
  CU(Error, cuEventRecord(ss->h_in.ready, s->h2d));

  // Kernel waits for H2D, then scatters into tile pool
  CU(Error, cuStreamWaitEvent(s->compute, ss->h_in.ready, 0));
  CU(Error, cuEventRecord(ss->t_scatter_start, s->compute));
  switch (bpe) {
    case 1:
      transpose_u8_v0((CUdeviceptr)pool,
                      (CUdeviceptr)pool + s->layout.tile_pool_bytes,
                      (CUdeviceptr)ss->d_in.data,
                      (CUdeviceptr)ss->d_in.data + s->stage.fill,
                      s->cursor,
                      s->layout.lifted_rank,
                      s->layout.d_lifted_shape,
                      s->layout.d_lifted_strides,
                      s->compute);
      break;
    case 2:
      transpose_u16_v0((CUdeviceptr)pool,
                       (CUdeviceptr)pool + s->layout.tile_pool_bytes,
                       (CUdeviceptr)ss->d_in.data,
                       (CUdeviceptr)ss->d_in.data + s->stage.fill,
                       s->cursor,
                       s->layout.lifted_rank,
                       s->layout.d_lifted_shape,
                       s->layout.d_lifted_strides,
                       s->compute);
      break;
    case 4:
      transpose_u32_v0((CUdeviceptr)pool,
                       (CUdeviceptr)pool + s->layout.tile_pool_bytes,
                       (CUdeviceptr)ss->d_in.data,
                       (CUdeviceptr)ss->d_in.data + s->stage.fill,
                       s->cursor,
                       s->layout.lifted_rank,
                       s->layout.d_lifted_shape,
                       s->layout.d_lifted_strides,
                       s->compute);
      break;
    case 8:
      transpose_u64_v0((CUdeviceptr)pool,
                       (CUdeviceptr)pool + s->layout.tile_pool_bytes,
                       (CUdeviceptr)ss->d_in.data,
                       (CUdeviceptr)ss->d_in.data + s->stage.fill,
                       s->cursor,
                       s->layout.lifted_rank,
                       s->layout.d_lifted_shape,
                       s->layout.d_lifted_strides,
                       s->compute);
      break;
    default:
      log_error("dispatch_scatter: unsupported bytes_per_element=%zu", bpe);
      goto Error;
  }
  CU(Error, cuEventRecord(ss->t_scatter_end, s->compute));
  // Record pool ready on whichever pool we're writing to
  if (s->pool_current == 0) {
    CU(Error, cuEventRecord(s->pool_A.ready, s->compute));
  } else {
    CU(Error, cuEventRecord(s->pool_B.ready, s->compute));
  }

  // LOD scatter: convert input to Morton+batch order alongside transpose
  if (s->num_levels > 0) {
    // Compute spatial size for LOD epoch offset
    uint64_t spatial_size = 1;
    for (int d = 1; d < s->config.rank; ++d)
      spatial_size *= s->config.dimensions[d].size;
    uint64_t lod_epoch_elements = s->lod_dim0 * spatial_size;
    uint64_t epoch_offset = (s->cursor % lod_epoch_elements);
    lod_scatter(
      (CUdeviceptr)s->d_morton_values.data,
      (CUdeviceptr)ss->d_in.data,
      &s->lod_plan,
      s->config.bytes_per_element,
      s->lod_dim0,
      elements,
      epoch_offset,
      s->compute);
  }

  s->cursor += elements;
  s->stage.current ^= 1;
  return 0;

Error:
  return 1;
}

struct writer_result
writer_append(struct writer* w, struct slice data)
{
  return w->append(w, data);
}

struct writer_result
writer_flush(struct writer* w)
{
  return w->flush(w);
}

struct writer_result
writer_append_wait(struct writer* w, struct slice data)
{
  int stalls = 0;
  const int max_stalls = 10;

  while (data.beg < data.end) {
    struct writer_result r = writer_append(w, data);
    if (r.error)
      return r;

    if (r.rest.beg == data.beg) {
      if (++stalls >= max_stalls) {
        log_error("writer_append_wait: no progress after %d retries", stalls);
        return writer_error_at(data.beg, data.end);
      }
      log_warn(
        "writer_append_wait: stall %d/%d, backing off", stalls, max_stalls);
      platform_sleep_ns(1000000LL << (stalls < 6 ? stalls : 6)); // 1ms..64ms
    } else {
      stalls = 0;
    }

    data = r.rest;
  }

  return writer_ok();
}

// Software CRC32C (Castagnoli) computed at runtime via a generated table.
static uint32_t crc32c_table[256];
static int crc32c_table_ready;

static void
crc32c_init_table(void)
{
  if (crc32c_table_ready)
    return;
  for (int i = 0; i < 256; ++i) {
    uint32_t crc = (uint32_t)i;
    for (int j = 0; j < 8; ++j)
      crc = (crc >> 1) ^ (0x82F63B78 & (0u - (crc & 1)));
    crc32c_table[i] = crc;
  }
  crc32c_table_ready = 1;
}

static uint32_t
crc32c(const void* data, size_t len)
{
  uint32_t crc = 0xFFFFFFFF;
  const uint8_t* p = (const uint8_t*)data;
  for (size_t i = 0; i < len; ++i)
    crc = crc32c_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFF;
}

// --- Unified shard delivery ---

// Emit completed shards (write index block + finalize).
// Works for both L0 and LOD levels.
static int
emit_shards(struct shard_state* ss)
{
  for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
    struct active_shard* sh = &ss->shards[si];
    if (!sh->writer)
      continue;

    size_t index_data_bytes = ss->tiles_per_shard_total * 2 * sizeof(uint64_t);
    size_t index_total_bytes = index_data_bytes + 4;
    uint8_t* index_buf = (uint8_t*)malloc(index_total_bytes);
    CHECK(Error, index_buf);

    memcpy(index_buf, sh->index, index_data_bytes);

    uint32_t crc_val = crc32c(index_buf, index_data_bytes);
    memcpy(index_buf + index_data_bytes, &crc_val, 4);

    int wrc = sh->writer->write(
      sh->writer, sh->data_cursor, index_buf, index_buf + index_total_bytes);
    free(index_buf);
    CHECK(Error, wrc == 0);

    CHECK(Error, sh->writer->finalize(sh->writer) == 0);

    sh->writer = NULL;
    sh->data_cursor = 0;
    memset(sh->index, 0xFF, ss->tiles_per_shard_total * 2 * sizeof(uint64_t));
  }

  ss->epoch_in_shard = 0;
  ss->shard_epoch++;
  return 0;

Error:
  return 1;
}

// Deliver compressed tile data from an aggregate slot to shards.
// level: 0 for L0, 1+ for LOD levels.
static int
deliver_to_shards(struct transpose_stream* s,
                  uint8_t level,
                  struct shard_state* ss,
                  struct aggregate_slot* agg_slot)
{
  const uint64_t tps_inner = ss->tiles_per_shard_inner;
  const uint64_t epoch_in_shard = ss->epoch_in_shard;

  for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
    uint64_t j_start = si * tps_inner;
    uint64_t j_end = j_start + tps_inner;

    struct active_shard* sh = &ss->shards[si];

    if (!sh->writer) {
      uint64_t flat = ss->shard_epoch * ss->shard_inner_count + si;
      sh->writer = s->config.shard_sink->open(s->config.shard_sink, level, flat);
      CHECK(Error, sh->writer);
    }

    size_t shard_bytes = agg_slot->h_offsets[j_end] - agg_slot->h_offsets[j_start];
    if (shard_bytes > 0) {
      const void* src =
        (const char*)agg_slot->h_aggregated + agg_slot->h_offsets[j_start];
      CHECK(Error,
            sh->writer->write(sh->writer,
                              sh->data_cursor,
                              src,
                              (const char*)src + shard_bytes) == 0);
    }

    for (uint64_t j = j_start; j < j_end; ++j) {
      size_t tile_size = agg_slot->h_offsets[j + 1] - agg_slot->h_offsets[j];
      if (tile_size > 0) {
        uint64_t within_inner = j - j_start;
        uint64_t slot = epoch_in_shard * tps_inner + within_inner;
        size_t tile_off =
          sh->data_cursor + (agg_slot->h_offsets[j] - agg_slot->h_offsets[j_start]);
        sh->index[2 * slot] = tile_off;
        sh->index[2 * slot + 1] = tile_size;
      }
    }
    sh->data_cursor += shard_bytes;
  }

  ss->epoch_in_shard++;

  if (ss->epoch_in_shard >= ss->tiles_per_shard_0)
    return emit_shards(ss);

  return 0;

Error:
  return 1;
}

// Wait for D2H on the given flush slot, record timing, deliver to sinks.
static struct writer_result
wait_and_deliver(struct transpose_stream* s, int fc)
{
  struct flush_slot* fs = &s->flush[fc];

  if (s->config.compress && s->config.shard_sink) {
    CU(Error, cuEventSynchronize(fs->ready));

    if (fs->lod_fired)
      accumulate_metric_cu(&s->metrics.lod, fs->t_lod_start,
                           fs->t_lod_end);
    accumulate_metric_cu(&s->metrics.compress, fs->t_compress_start,
                         s->flush[fc].d_compressed.ready);
    accumulate_metric_cu(&s->metrics.d2h, fs->t_d2h_start, fs->ready);

    // Deliver each firing level
    for (int i = 0; i < fs->num_firing; ++i) {
      uint8_t level = fs->firing_levels[i];
      if (level == 0) {
        // L0
        struct aggregate_slot* agg = &s->agg[fc];
        if (deliver_to_shards(s, 0, &s->shard, agg))
          goto Error;
      } else {
        // LOD level
        struct level_state* lev = &s->levels[level - 1];
        struct aggregate_slot* agg = &lev->agg[fc];
        if (deliver_to_shards(s, level, &lev->shard, agg))
          goto Error;
      }
    }
  } else {
    // Uncompressed path: wait for host pool ready
    if (fc == 0) {
      CU(Error, cuEventSynchronize(s->pool_A_host.ready));
      accumulate_metric_cu(&s->metrics.d2h, fs->t_d2h_start,
                           s->pool_A_host.ready);
      if (s->config.sink) {
        struct slice tiles = {
          .beg = s->pool_A_host.data,
          .end = (char*)s->pool_A_host.data + s->layout.tile_pool_bytes,
        };
        return writer_append_wait(s->config.sink, tiles);
      }
    } else {
      CU(Error, cuEventSynchronize(s->pool_B_host.ready));
      accumulate_metric_cu(&s->metrics.d2h, fs->t_d2h_start,
                           s->pool_B_host.ready);
      if (s->config.sink) {
        struct slice tiles = {
          .beg = s->pool_B_host.data,
          .end = (char*)s->pool_B_host.data + s->layout.tile_pool_bytes,
        };
        return writer_append_wait(s->config.sink, tiles);
      }
    }
  }

  return writer_ok();

Error:
  return writer_error();
}

// Drain pending flush from the previous epoch.
static struct writer_result
drain_pending_flush(struct transpose_stream* s)
{
  if (!s->flush_pending)
    return writer_ok();

  s->flush_pending = 0;
  return wait_and_deliver(s, s->flush_current);
}

// Forward declaration for fire_lod
static int
kick_epoch(struct transpose_stream* s, int fc, uint64_t ptr_offset,
           uint64_t num_chunks, const uint8_t* firing_levels, int num_firing);

// --- LOD fire (deep Morton buffer approach) ---

// Fire LOD cascade after a full LOD epoch (or partial at flush).
// actual_dim0: number of dim-0 elements accumulated in the Morton buffer
//   (lod_dim0 for full epoch, less for partial).
// Writes firing level indices (1-based) into out_firing.
// Returns number of firing LOD levels, or -1 on error.
static int
fire_lod(struct transpose_stream* s, uint64_t actual_dim0,
         uint8_t* out_firing)
{
  const size_t bpe = s->config.bytes_per_element;
  const int dim0_ds = s->config.dimensions[0].downsample;
  const uint64_t tile_size_0 = s->config.dimensions[0].tile_size;
  const struct lod_plan* plan = &s->lod_plan;

  // vals_per_slice: stride between dim-0 slices in the full Morton buffer
  uint64_t vals_per_slice = plan->batch_level_ends[plan->nlev - 1];

  // 1. Spatial reduce: produce levels 1..nlev-1 from level 0
  lod_reduce(
    (CUdeviceptr)s->d_morton_values.data,
    plan, bpe, actual_dim0, s->compute);

  // 2. For each LOD level: dim-0 reduce, then emit tile epochs in rounds
  // First pass: compute effective_dim0 per level and do dim-0 reduction
  uint64_t eff_dim0[MAX_LOD_LEVELS];
  for (int k = 0; k < s->num_levels; ++k) {
    int lv = k + 1; // plan level index (1-based)

    // Morton level data pointer
    uint64_t morton_level_offset = plan->batch_level_ends[k];
    CUdeviceptr morton_level_ptr =
      (CUdeviceptr)s->d_morton_values.data +
      morton_level_offset * bpe;
    // Note: the above points to the level's data at dim0_coord=0.
    // With the stacked layout, level k data for dim0_coord d is at:
    //   base + d * vals_per_slice * bpe

    // Dim-0 reduce: halve k times (if dim0 is downsampled)
    uint64_t current_dim0 = actual_dim0;
    if (dim0_ds) {
      for (int step = 0; step < lv && current_dim0 >= 2; ++step) {
        // vals_per_slice for this level within the full buffer:
        // Each dim-0 slice has vals_per_slice elements total, but we only
        // operate on the portion belonging to this level.
        // The level-k data sits at offset morton_level_offset within each
        // dim-0 slice, and has ds_counts[lv] * batch_count elements.
        // But reduce_dim0 operates on the full slice stride — it reads
        // d_values[slice * vals_per_slice + elem] so we need to pass
        // vals_per_slice as the stride.
        uint64_t level_count = plan->batch_count * plan->ds_counts[lv];
        lod_reduce_dim0(
          morton_level_ptr, bpe, current_dim0, level_count,
          vals_per_slice, s->compute);
        current_dim0 /= 2;
      }
    }
    eff_dim0[k] = current_dim0;
  }

  // 3. Emit tile epochs in rounds
  // max_tile_epochs comes from the shallowest LOD level (most epochs)
  uint64_t max_tile_epochs = 0;
  for (int k = 0; k < s->num_levels; ++k) {
    uint64_t te = eff_dim0[k] / tile_size_0;
    if (te < 1) te = 1;
    if (te > max_tile_epochs) max_tile_epochs = te;
  }

  int total_fired = 0;
  for (uint64_t e = 0; e < max_tile_epochs; ++e) {
    uint8_t round_firing[MAX_LOD_LEVELS];
    int round_count = 0;
    uint64_t round_tiles = 0;

    for (int k = 0; k < s->num_levels; ++k) {
      uint64_t k_epochs = eff_dim0[k] / tile_size_0;
      if (k_epochs < 1) k_epochs = 1;
      if (e >= k_epochs) continue;

      int lv = k + 1;
      struct level_state* lev = &s->levels[k];

      // Compute source pointer: level k in Morton buffer, at dim-0 offset e * tile_size_0
      uint64_t morton_level_offset = plan->batch_level_ends[k];
      CUdeviceptr morton_level_ptr =
        (CUdeviceptr)s->d_morton_values.data +
        morton_level_offset * bpe;
      CUdeviceptr src = morton_level_ptr +
        e * tile_size_0 * vals_per_slice * bpe;

      // Scatter to tile pool
      CUdeviceptr dst = (CUdeviceptr)s->pool_B.data + s->level_offset[k + 1];
      lod_to_tiles(
        dst, src,
        plan, lv, bpe, tile_size_0, vals_per_slice,
        lev->layout.lifted_rank,
        lev->layout.d_lifted_shape,
        lev->layout.d_lifted_strides,
        s->compute);

      round_firing[round_count++] = (uint8_t)lv;
      round_tiles += lev->layout.slot_count;
    }

    if (round_count > 0) {
      CU(Error, cuEventRecord(s->pool_B.ready, s->compute));

      uint64_t M0 = s->layout.slot_count;
      if (kick_epoch(s, 1 /*fc=B*/, M0, round_tiles, round_firing, round_count))
        return -1;
      struct writer_result wr = wait_and_deliver(s, 1);
      if (wr.error)
        return -1;

      // Record which levels fired (for caller, deduplicated)
      for (int i = 0; i < round_count; ++i) {
        int found = 0;
        for (int j = 0; j < total_fired; ++j)
          if (out_firing[j] == round_firing[i]) { found = 1; break; }
        if (!found)
          out_firing[total_fired++] = round_firing[i];
      }
    }
  }

  return total_fired;

Error:
  return -1;
}

// --- Epoch flush pipeline ---

// Kick compress + aggregate + D2H for the current epoch.
// fc: flush slot index (0 or 1, matches pool_current before swap).
// ptr_offset: offset into pointer arrays (skip L0 pointers for LOD-only kicks).
// num_chunks: number of tiles to compress.
// firing_levels[]: array of level indices that fired.
// num_firing: length of firing_levels.
static int
kick_epoch(struct transpose_stream* s,
           int fc,
           uint64_t ptr_offset,
           uint64_t num_chunks,
           const uint8_t* firing_levels,
           int num_firing)
{
  struct flush_slot* fs = &s->flush[fc];
  fs->num_firing = num_firing;
  memcpy(fs->firing_levels, firing_levels, num_firing);

  if (s->config.compress && s->config.shard_sink) {
    // Wait for scatter to finish
    if (fc == 0) {
      CU(Error, cuStreamWaitEvent(s->compress, s->pool_A.ready, 0));
    } else {
      CU(Error, cuStreamWaitEvent(s->compress, s->pool_B.ready, 0));
    }

    // Offset pointer arrays for LOD-only kicks
    void** uncomp_ptrs = (void**)((char*)fs->d_uncomp_ptrs + ptr_offset * sizeof(void*));
    void** comp_ptrs = (void**)((char*)fs->d_comp_ptrs + ptr_offset * sizeof(void*));

    CU(Error, cuEventRecord(fs->t_compress_start, s->compress));
    CHECK(Error,
          compress_batch_async(
            (const void* const*)uncomp_ptrs,
            s->d_uncomp_sizes,
            s->layout.tile_stride * s->config.bytes_per_element,
            num_chunks,
            s->d_comp_temp,
            s->comp_temp_bytes,
            comp_ptrs,
            s->d_comp_sizes + ptr_offset,
            s->compress) == 0);

    // nvcomp may use internal CUDA operations not captured by stream events.
    // Synchronize before reading compressed output.
    CU(Error, cuStreamSynchronize(s->compress));
    CU(Error, cuEventRecord(fs->d_compressed.ready, s->compress));

    // Per-level aggregate + D2H
    uint64_t tile_offset = 0;
    CU(Error, cuStreamWaitEvent(s->d2h, fs->d_compressed.ready, 0));
    CU(Error, cuEventRecord(fs->t_d2h_start, s->d2h));

    for (int i = 0; i < num_firing; ++i) {
      uint8_t level = firing_levels[i];
      uint64_t M;
      struct aggregate_layout* al;
      struct aggregate_slot* agg;
      size_t level_comp_pool_bytes;

      if (level == 0) {
        M = s->layout.slot_count;
        al = &s->agg_layout;
        agg = &s->agg[fc];
        level_comp_pool_bytes = s->comp_pool_bytes;
      } else {
        struct level_state* lev = &s->levels[level - 1];
        M = lev->layout.slot_count;
        al = &lev->agg_layout;
        agg = &lev->agg[fc];
        level_comp_pool_bytes = M * s->max_comp_chunk_bytes;
      }

      // d_compressed base for this level (accounting for ptr_offset)
      void* d_comp_base =
        (char*)fs->d_compressed.data + (ptr_offset + tile_offset) * s->max_comp_chunk_bytes;
      size_t* d_sizes_base = s->d_comp_sizes + ptr_offset + tile_offset;

      CHECK(Error,
            aggregate_by_shard_async(al, d_comp_base, d_sizes_base, agg,
                                     s->compress) == 0);
      CU(Error, cuEventRecord(agg->ready, s->compress));

      // D2H: aggregated data + offsets
      CU(Error, cuStreamWaitEvent(s->d2h, agg->ready, 0));
      CU(Error,
         cuMemcpyDtoHAsync(agg->h_aggregated,
                           (CUdeviceptr)agg->d_aggregated,
                           level_comp_pool_bytes,
                           s->d2h));
      CU(Error,
         cuMemcpyDtoHAsync(agg->h_offsets,
                           (CUdeviceptr)agg->d_offsets,
                           (al->covering_count + 1) * sizeof(size_t),
                           s->d2h));
      CU(Error, cuEventRecord(agg->ready, s->d2h));

      tile_offset += M;
    }

    CU(Error, cuEventRecord(fs->ready, s->d2h));
  } else {
    // Uncompressed path: D2H the L0 pool
    CU(Error, cuEventRecord(fs->t_d2h_start, s->d2h));
    if (fc == 0) {
      CU(Error, cuStreamWaitEvent(s->d2h, s->pool_A.ready, 0));
      CU(Error,
         cuMemcpyDtoHAsync(s->pool_A_host.data,
                           (CUdeviceptr)s->pool_A.data,
                           s->layout.tile_pool_bytes,
                           s->d2h));
      CU(Error, cuEventRecord(s->pool_A_host.ready, s->d2h));
    } else {
      CU(Error, cuStreamWaitEvent(s->d2h, s->pool_B.ready, 0));
      CU(Error,
         cuMemcpyDtoHAsync(s->pool_B_host.data,
                           (CUdeviceptr)s->pool_B.data,
                           s->layout.tile_pool_bytes,
                           s->d2h));
      CU(Error, cuEventRecord(s->pool_B_host.ready, s->d2h));
    }
    CU(Error, cuEventRecord(fs->ready, s->d2h));
  }

  return 0;

Error:
  return 1;
}

// Flush the current epoch's tile pool: cascade LODs, compress, D2H, swap.
static struct writer_result
flush_epoch(struct transpose_stream* s)
{
  const int fc = s->pool_current; // 0=A, 1=B

  // Deliver the previous epoch if its D2H is still in flight
  struct writer_result r = drain_pending_flush(s);
  if (r.error)
    return r;

  // Build list of firing levels
  uint8_t firing[MAX_LOD_LEVELS + 1];
  int num_firing = 0;
  firing[num_firing++] = 0; // L0 always fires
  s->flush[fc].lod_fired = 0;

  // LOD: increment counter, fire if full LOD epoch
  if (s->num_levels > 0) {
    s->lod_epoch_counter++;

    if (s->lod_epoch_counter >= s->lod_epoch_period) {
      // Full LOD epoch ready — fire before kicking L0
      CU(Error, cuEventRecord(s->flush[fc].t_lod_start, s->compute));
      uint8_t lod_firing[MAX_LOD_LEVELS];
      int num_lod_fired = fire_lod(s, s->lod_dim0, lod_firing);
      if (num_lod_fired < 0)
        return writer_error();
      CU(Error, cuEventRecord(s->flush[fc].t_lod_end, s->compute));
      s->flush[fc].lod_fired = 1;

      // Zero Morton buffer for next LOD epoch
      size_t morton_buf_bytes = s->lod_dim0 *
        s->lod_plan.batch_level_ends[s->lod_plan.nlev - 1] *
        s->config.bytes_per_element;
      CU(Error,
         cuMemsetD8Async((CUdeviceptr)s->d_morton_values.data, 0,
                         morton_buf_bytes, s->compute));

      s->lod_epoch_counter = 0;
    }
  }

  // Kick L0 only (LOD levels are kicked inside fire_lod)
  if (kick_epoch(s, fc, 0, s->layout.slot_count, firing, num_firing))
    return writer_error();

  // Swap pool and zero next L0 region
  s->pool_current ^= 1;
  void* next = current_pool(s);
  CU(Error,
     cuMemsetD8Async((CUdeviceptr)next, 0, s->layout.tile_pool_bytes,
                     s->compute));

  s->flush_pending = 1;
  s->flush_current = fc;
  return writer_ok();

Error:
  return writer_error();
}

// Synchronously flush the current tile pool (used for the final partial epoch).
static struct writer_result
flush_epoch_sync(struct transpose_stream* s)
{
  const int fc = s->pool_current;

  // Only L0 fires for sync flush (no LOD cascade for partial epochs)
  uint8_t firing[1] = { 0 };
  if (kick_epoch(s, fc, 0, s->layout.slot_count, firing, 1))
    return writer_error();
  return wait_and_deliver(s, fc);
}

struct stream_metrics
transpose_stream_get_metrics(const struct transpose_stream* s)
{
  return s->metrics;
}

// --- Destroy ---

void
transpose_stream_destroy(struct transpose_stream* stream)
{
  if (!stream)
    return;

  CUWARN(cuStreamDestroy(stream->h2d));
  CUWARN(cuStreamDestroy(stream->compute));
  CUWARN(cuStreamDestroy(stream->compress));
  CUWARN(cuStreamDestroy(stream->d2h));

  CUWARN(cuMemFree((CUdeviceptr)stream->layout.d_lifted_shape));
  CUWARN(cuMemFree((CUdeviceptr)stream->layout.d_lifted_strides));

  for (int i = 0; i < 2; ++i) {
    struct staging_slot* ss = &stream->stage.slot[i];
    CUWARN(cuEventDestroy(ss->t_h2d_start));
    CUWARN(cuEventDestroy(ss->t_scatter_start));
    CUWARN(cuEventDestroy(ss->t_scatter_end));
    buffer_free(&ss->h_in);
    buffer_free(&ss->d_in);
  }

  // Tile pools
  buffer_free(&stream->pool_A);
  buffer_free(&stream->pool_B);
  buffer_free(&stream->pool_A_host);
  buffer_free(&stream->pool_B_host);

  // Flush slots
  for (int i = 0; i < 2; ++i) {
    struct flush_slot* fs = &stream->flush[i];
    buffer_free(&fs->d_compressed);
    device_free(fs->d_uncomp_ptrs);
    device_free(fs->d_comp_ptrs);
    CUWARN(cuEventDestroy(fs->t_compress_start));
    CUWARN(cuEventDestroy(fs->t_d2h_start));
    CUWARN(cuEventDestroy(fs->t_lod_start));
    CUWARN(cuEventDestroy(fs->t_lod_end));
    CUWARN(cuEventDestroy(fs->ready));
  }

  // Shared compress state
  device_free(stream->d_comp_sizes);
  device_free(stream->d_uncomp_sizes);
  device_free(stream->d_comp_temp);

  // L0 aggregate + shard
  aggregate_layout_destroy(&stream->agg_layout);
  for (int i = 0; i < 2; ++i)
    aggregate_slot_destroy(&stream->agg[i]);

  if (stream->shard.shards) {
    for (uint64_t i = 0; i < stream->shard.shard_inner_count; ++i)
      free(stream->shard.shards[i].index);
    free(stream->shard.shards);
  }

  // LOD levels
  for (int lv = 0; lv < stream->num_levels; ++lv) {
    struct level_state* lev = &stream->levels[lv];

    device_free(lev->layout.d_lifted_shape);
    device_free(lev->layout.d_lifted_strides);

    aggregate_layout_destroy(&lev->agg_layout);
    for (int i = 0; i < 2; ++i)
      aggregate_slot_destroy(&lev->agg[i]);

    if (lev->shard.shards) {
      for (uint64_t i = 0; i < lev->shard.shard_inner_count; ++i)
        free(lev->shard.shards[i].index);
      free(lev->shard.shards);
    }
  }

  // LOD plan + Morton buffer
  lod_plan_destroy(&stream->lod_plan);
  buffer_free(&stream->d_morton_values);

  *stream = (struct transpose_stream){ 0 };
}

// --- Create ---

// Forward declarations for vtable
static struct writer_result
transpose_stream_append(struct writer* self, struct slice input);
static struct writer_result
transpose_stream_flush(struct writer* self);

static int
init_cuda_streams_and_events(struct transpose_stream* s)
{
  CU(Fail, cuStreamCreate(&s->h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&s->compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&s->compress, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&s->d2h, CU_STREAM_NON_BLOCKING));

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventCreate(&s->stage.slot[i].t_h2d_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&s->stage.slot[i].t_scatter_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&s->stage.slot[i].t_scatter_end, CU_EVENT_DEFAULT));
  }

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventCreate(&s->flush[i].t_compress_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&s->flush[i].t_d2h_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&s->flush[i].t_lod_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&s->flush[i].t_lod_end, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&s->flush[i].ready, CU_EVENT_DEFAULT));
  }

  return 0;
Fail:
  return 1;
}

static int
init_l0_layout(struct transpose_stream* s)
{
  const uint8_t rank = s->config.rank;
  const size_t bpe = s->config.bytes_per_element;
  const struct dimension* dims = s->config.dimensions;

  s->layout.lifted_rank = 2 * rank;
  s->layout.tile_elements = 1;

  uint64_t tile_count[MAX_RANK];
  for (int i = 0; i < rank; ++i) {
    tile_count[i] = ceildiv(dims[i].size, dims[i].tile_size);
    s->layout.lifted_shape[2 * i] = tile_count[i];
    s->layout.lifted_shape[2 * i + 1] = dims[i].tile_size;
    s->layout.tile_elements *= dims[i].tile_size;
  }

  {
    size_t alignment =
      s->config.compress ? compress_get_input_alignment() : 1;
    size_t tile_bytes = s->layout.tile_elements * bpe;
    size_t padded_bytes = align_up(tile_bytes, alignment);
    s->layout.tile_stride = padded_bytes / bpe;
  }

  {
    int64_t n_stride = 1;
    int64_t t_stride = (int64_t)s->layout.tile_stride;

    for (int i = rank - 1; i >= 0; --i) {
      s->layout.lifted_strides[2 * i + 1] = n_stride;
      n_stride *= (int64_t)dims[i].tile_size;

      s->layout.lifted_strides[2 * i] = t_stride;
      t_stride *= (int64_t)tile_count[i];
    }
  }

  s->layout.slot_count =
    s->layout.lifted_strides[0] / s->layout.tile_stride;
  s->layout.epoch_elements =
    s->layout.slot_count * s->layout.tile_elements;
  s->layout.lifted_strides[0] = 0; // collapse epoch dim
  s->layout.tile_pool_bytes =
    s->layout.slot_count * s->layout.tile_stride * bpe;

  {
    const size_t shape_bytes = s->layout.lifted_rank * sizeof(uint64_t);
    const size_t strides_bytes = s->layout.lifted_rank * sizeof(int64_t);
    CU(Fail, cuMemAlloc((CUdeviceptr*)&s->layout.d_lifted_shape, shape_bytes));
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&s->layout.d_lifted_strides, strides_bytes));
    CU(Fail, cuMemcpyHtoD((CUdeviceptr)s->layout.d_lifted_shape,
                           s->layout.lifted_shape, shape_bytes));
    CU(Fail, cuMemcpyHtoD((CUdeviceptr)s->layout.d_lifted_strides,
                           s->layout.lifted_strides, strides_bytes));
  }

  return 0;
Fail:
  return 1;
}

static int
init_staging_buffers(struct transpose_stream* s)
{
  for (int i = 0; i < 2; ++i) {
    CHECK(Fail,
          (s->stage.slot[i].h_in =
             buffer_new(s->config.buffer_capacity_bytes, host, 0))
            .data);
    CHECK(Fail,
          (s->stage.slot[i].d_in =
             buffer_new(s->config.buffer_capacity_bytes, device, 0))
            .data);
  }

  return 0;
Fail:
  return 1;
}

static int
init_lod_plan(struct transpose_stream* s)
{
  const uint8_t rank = s->config.rank;
  const size_t bpe = s->config.bytes_per_element;
  const struct dimension* dims = s->config.dimensions;
  const uint64_t M0 = s->layout.slot_count;

  s->M_total = M0;
  s->level_offset[0] = 0;

  int num_levels = 0;

  uint8_t ds_mask = 0;
  uint8_t spatial_ds_mask = 0;
  if (s->config.enable_lod && s->config.compress && s->config.shard_sink) {
    for (int d = 0; d < rank; ++d) {
      if (dims[d].downsample)
        ds_mask |= (1 << d);
    }
    for (int d = 1; d < rank; ++d) {
      if (dims[d].downsample)
        spatial_ds_mask |= (1 << (d - 1));
    }
  }

  if (ds_mask) {
    struct dimension test_dims[MAX_RANK / 2];
    for (int d = 0; d < rank; ++d)
      test_dims[d] = dims[d];

    for (int lv = 0; lv < MAX_LOD_LEVELS; ++lv) {
      struct dimension next[MAX_RANK / 2];
      int all_can_continue = 1;
      for (int d = 0; d < rank; ++d) {
        next[d] = test_dims[d];
        if (test_dims[d].downsample)
          next[d].size = ceildiv(test_dims[d].size, 2);
        if (next[d].downsample && next[d].size <= next[d].tile_size)
          all_can_continue = 0;
      }
      num_levels++;

      if (!all_can_continue)
        break;

      for (int d = 0; d < rank; ++d)
        test_dims[d] = next[d];
    }
  }

  s->num_levels = num_levels;

  if (num_levels > 0 && spatial_ds_mask) {
    uint64_t spatial_shape[MAX_RANK / 2];
    for (int d = 1; d < rank; ++d)
      spatial_shape[d - 1] = dims[d].size;
    CHECK(Fail,
          lod_plan_init(&s->lod_plan, rank - 1, spatial_shape,
                        spatial_ds_mask, num_levels + 1));

    int dim0_ds = (ds_mask & 1) ? 1 : 0;
    s->lod_dim0 = dim0_ds
      ? dims[0].tile_size << num_levels
      : dims[0].tile_size;
    s->lod_epoch_period = dim0_ds ? (1ull << num_levels) : 1;
    s->lod_epoch_counter = 0;

    size_t morton_buf_elements =
      s->lod_dim0 * s->lod_plan.batch_level_ends[s->lod_plan.nlev - 1];
    size_t morton_buf_bytes = morton_buf_elements * bpe;
    CHECK(Fail,
          (s->d_morton_values = buffer_new(morton_buf_bytes, device, 0)).data);
    CU(Fail, cuMemsetD8Async((CUdeviceptr)s->d_morton_values.data, 0,
                             morton_buf_bytes, s->compute));
  }

  return 0;
Fail:
  return 1;
}

static int
init_level_states(struct transpose_stream* s)
{
  const uint8_t rank = s->config.rank;
  const size_t bpe = s->config.bytes_per_element;
  const struct dimension* dims = s->config.dimensions;
  const int num_levels = s->num_levels;
  const uint64_t M0 = s->layout.slot_count;

  if (num_levels <= 0)
    return 0;

  const struct dimension* parent_dims = dims;

  for (int lv = 0; lv < num_levels; ++lv) {
    struct level_state* lev = &s->levels[lv];
    for (int d = 0; d < rank; ++d) {
      lev->dimensions[d] = parent_dims[d];
      if (parent_dims[d].downsample)
        lev->dimensions[d].size = ceildiv(parent_dims[d].size, 2);
    }

    uint64_t lod_tile_count[MAX_RANK];
    uint64_t lod_tiles_per_shard[MAX_RANK];
    for (int d = 0; d < rank; ++d) {
      lod_tile_count[d] =
        ceildiv(lev->dimensions[d].size, lev->dimensions[d].tile_size);
      uint64_t tps = lev->dimensions[d].tiles_per_shard;
      lod_tiles_per_shard[d] = (tps == 0) ? lod_tile_count[d] : tps;
    }

    lev->layout.lifted_rank = 2 * rank;
    lev->layout.tile_elements = 1;
    for (int d = 0; d < rank; ++d) {
      lev->layout.lifted_shape[2 * d] = lod_tile_count[d];
      lev->layout.lifted_shape[2 * d + 1] = lev->dimensions[d].tile_size;
      lev->layout.tile_elements *= lev->dimensions[d].tile_size;
    }

    {
      size_t alignment =
        s->config.compress ? compress_get_input_alignment() : 1;
      size_t tile_bytes = lev->layout.tile_elements * bpe;
      size_t padded_bytes = align_up(tile_bytes, alignment);
      lev->layout.tile_stride = padded_bytes / bpe;
    }

    {
      int64_t n_stride = 1;
      int64_t t_stride = (int64_t)lev->layout.tile_stride;
      for (int d = rank - 1; d >= 0; --d) {
        lev->layout.lifted_strides[2 * d + 1] = n_stride;
        n_stride *= (int64_t)lev->dimensions[d].tile_size;
        lev->layout.lifted_strides[2 * d] = t_stride;
        t_stride *= (int64_t)lod_tile_count[d];
      }
    }

    lev->layout.slot_count =
      lev->layout.lifted_strides[0] / lev->layout.tile_stride;
    lev->layout.epoch_elements =
      lev->layout.slot_count * lev->layout.tile_elements;
    lev->layout.lifted_strides[0] = 0;
    lev->layout.tile_pool_bytes =
      lev->layout.slot_count * lev->layout.tile_stride * bpe;

    {
      const size_t shape_bytes = lev->layout.lifted_rank * sizeof(uint64_t);
      const size_t strides_bytes = lev->layout.lifted_rank * sizeof(int64_t);
      CU(Fail,
         cuMemAlloc((CUdeviceptr*)&lev->layout.d_lifted_shape, shape_bytes));
      CU(Fail, cuMemAlloc((CUdeviceptr*)&lev->layout.d_lifted_strides,
                           strides_bytes));
      CU(Fail, cuMemcpyHtoD((CUdeviceptr)lev->layout.d_lifted_shape,
                             lev->layout.lifted_shape, shape_bytes));
      CU(Fail, cuMemcpyHtoD((CUdeviceptr)lev->layout.d_lifted_strides,
                             lev->layout.lifted_strides, strides_bytes));
    }

    s->level_offset[lv + 1] =
      s->level_offset[lv] +
      (lv == 0 ? M0 : s->levels[lv - 1].layout.slot_count) *
        s->layout.tile_stride * bpe;
    s->M_total += lev->layout.slot_count;

    CHECK(Fail,
          aggregate_layout_init(&lev->agg_layout, rank, lod_tile_count,
                                lod_tiles_per_shard, lev->layout.slot_count,
                                0 /* filled below */) == 0);

    lev->shard.tiles_per_shard_0 = lod_tiles_per_shard[0];
    lev->shard.tiles_per_shard_inner = 1;
    for (int d = 1; d < rank; ++d)
      lev->shard.tiles_per_shard_inner *= lod_tiles_per_shard[d];
    lev->shard.tiles_per_shard_total =
      lev->shard.tiles_per_shard_0 * lev->shard.tiles_per_shard_inner;

    lev->shard.shard_inner_count = 1;
    for (int d = 1; d < rank; ++d)
      lev->shard.shard_inner_count *=
        ceildiv(lod_tile_count[d], lod_tiles_per_shard[d]);

    lev->shard.shards = (struct active_shard*)calloc(
      lev->shard.shard_inner_count, sizeof(struct active_shard));
    CHECK(Fail, lev->shard.shards);

    size_t index_bytes =
      2 * lev->shard.tiles_per_shard_total * sizeof(uint64_t);
    for (uint64_t i = 0; i < lev->shard.shard_inner_count; ++i) {
      lev->shard.shards[i].index = (uint64_t*)malloc(index_bytes);
      CHECK(Fail, lev->shard.shards[i].index);
      memset(lev->shard.shards[i].index, 0xFF, index_bytes);
    }

    lev->shard.epoch_in_shard = 0;
    lev->shard.shard_epoch = 0;

    parent_dims = lev->dimensions;
  }

  return 0;
Fail:
  return 1;
}

static int
init_tile_pools(struct transpose_stream* s)
{
  const size_t bpe = s->config.bytes_per_element;
  const uint64_t M0 = s->layout.slot_count;
  const size_t A_bytes = M0 * s->layout.tile_stride * bpe;
  const size_t B_bytes = s->M_total * s->layout.tile_stride * bpe;

  CHECK(Fail, (s->pool_A = buffer_new(A_bytes, device, 0)).data);
  CHECK(Fail, (s->pool_B = buffer_new(B_bytes, device, 0)).data);
  CU(Fail,
     cuMemsetD8Async((CUdeviceptr)s->pool_A.data, 0, A_bytes, s->compute));
  CU(Fail,
     cuMemsetD8Async((CUdeviceptr)s->pool_B.data, 0, B_bytes, s->compute));

  CHECK(Fail,
        (s->pool_A_host = buffer_new(s->layout.tile_pool_bytes, host, 0))
          .data);
  CHECK(Fail,
        (s->pool_B_host = buffer_new(s->layout.tile_pool_bytes, host, 0))
          .data);

  return 0;
Fail:
  return 1;
}

static int
init_compression(struct transpose_stream* s)
{
  if (!s->config.compress)
    return 0;

  const size_t bpe = s->config.bytes_per_element;
  const uint64_t M0 = s->layout.slot_count;
  const int num_levels = s->num_levels;
  const size_t tile_bytes = s->layout.tile_stride * bpe;

  s->max_comp_chunk_bytes = align_up(
    compress_get_max_output_size(tile_bytes), compress_get_input_alignment());
  CHECK(Fail, s->max_comp_chunk_bytes > 0);
  s->comp_pool_bytes = M0 * s->max_comp_chunk_bytes;

  s->comp_temp_bytes = compress_get_temp_size(s->M_total, tile_bytes);
  if (s->comp_temp_bytes > 0)
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&s->d_comp_temp, s->comp_temp_bytes));

  CU(Fail, cuMemAlloc((CUdeviceptr*)&s->d_comp_sizes,
                       s->M_total * sizeof(size_t)));

  {
    size_t* h_sizes = (size_t*)malloc(s->M_total * sizeof(size_t));
    CHECK(Fail, h_sizes);
    for (uint64_t k = 0; k < s->M_total; ++k)
      h_sizes[k] = tile_bytes;
    CU(Fail, cuMemAlloc((CUdeviceptr*)&s->d_uncomp_sizes,
                         s->M_total * sizeof(size_t)));
    CUresult rc = cuMemcpyHtoD((CUdeviceptr)s->d_uncomp_sizes, h_sizes,
                                s->M_total * sizeof(size_t));
    free(h_sizes);
    CU(Fail, rc);
  }

  for (int fc = 0; fc < 2; ++fc) {
    struct flush_slot* fs = &s->flush[fc];

    CHECK(Fail,
          (fs->d_compressed =
             buffer_new(s->M_total * s->max_comp_chunk_bytes, device, 0))
            .data);
    CU(Fail, cuMemAlloc((CUdeviceptr*)&fs->d_uncomp_ptrs,
                         s->M_total * sizeof(void*)));
    CU(Fail, cuMemAlloc((CUdeviceptr*)&fs->d_comp_ptrs,
                         s->M_total * sizeof(void*)));

    void** h_ptrs = (void**)malloc(s->M_total * sizeof(void*));
    CHECK(Fail, h_ptrs);

    if (fc == 0) {
      for (uint64_t k = 0; k < M0; ++k)
        h_ptrs[k] = (char*)s->pool_A.data + k * tile_bytes;
      uint64_t off = M0;
      for (int lv = 0; lv < num_levels; ++lv) {
        uint64_t Mk = s->levels[lv].layout.slot_count;
        size_t base = s->level_offset[lv + 1];
        for (uint64_t k = 0; k < Mk; ++k)
          h_ptrs[off + k] = (char*)s->pool_B.data + base + k * tile_bytes;
        off += Mk;
      }
    } else {
      for (uint64_t k = 0; k < s->M_total; ++k)
        h_ptrs[k] = (char*)s->pool_B.data + k * tile_bytes;
    }
    CU(Fail, cuMemcpyHtoD((CUdeviceptr)fs->d_uncomp_ptrs, h_ptrs,
                            s->M_total * sizeof(void*)));

    for (uint64_t k = 0; k < s->M_total; ++k)
      h_ptrs[k] =
        (char*)fs->d_compressed.data + k * s->max_comp_chunk_bytes;
    CU(Fail, cuMemcpyHtoD((CUdeviceptr)fs->d_comp_ptrs, h_ptrs,
                            s->M_total * sizeof(void*)));
    free(h_ptrs);
  }

  return 0;
Fail:
  return 1;
}

static int
init_l0_aggregate_and_shards(struct transpose_stream* s)
{
  if (!s->config.compress || !s->config.shard_sink)
    return 0;

  const uint8_t rank = s->config.rank;
  const struct dimension* dims = s->config.dimensions;
  const int num_levels = s->num_levels;

  crc32c_init_table();

  uint64_t tile_count[MAX_RANK];
  uint64_t tiles_per_shard[MAX_RANK];
  for (int d = 0; d < rank; ++d) {
    tile_count[d] = ceildiv(dims[d].size, dims[d].tile_size);
    uint64_t tps = dims[d].tiles_per_shard;
    tiles_per_shard[d] = (tps == 0) ? tile_count[d] : tps;
  }

  CHECK(Fail,
        aggregate_layout_init(&s->agg_layout, rank, tile_count,
                              tiles_per_shard, s->layout.slot_count,
                              s->max_comp_chunk_bytes) == 0);

  for (int i = 0; i < 2; ++i)
    CHECK(Fail,
          aggregate_slot_init(&s->agg[i], &s->agg_layout,
                              s->comp_pool_bytes) == 0);

  s->shard.tiles_per_shard_0 = tiles_per_shard[0];
  s->shard.tiles_per_shard_inner = 1;
  for (int d = 1; d < rank; ++d)
    s->shard.tiles_per_shard_inner *= tiles_per_shard[d];
  s->shard.tiles_per_shard_total =
    s->shard.tiles_per_shard_0 * s->shard.tiles_per_shard_inner;

  s->shard.shard_inner_count = 1;
  for (int d = 1; d < rank; ++d)
    s->shard.shard_inner_count *= ceildiv(tile_count[d], tiles_per_shard[d]);

  s->shard.shards = (struct active_shard*)calloc(
    s->shard.shard_inner_count, sizeof(struct active_shard));
  CHECK(Fail, s->shard.shards);

  size_t index_bytes = 2 * s->shard.tiles_per_shard_total * sizeof(uint64_t);
  for (uint64_t i = 0; i < s->shard.shard_inner_count; ++i) {
    s->shard.shards[i].index = (uint64_t*)malloc(index_bytes);
    CHECK(Fail, s->shard.shards[i].index);
    memset(s->shard.shards[i].index, 0xFF, index_bytes);
  }

  s->shard.epoch_in_shard = 0;
  s->shard.shard_epoch = 0;

  for (int i = 0; i < 2; ++i)
    CU(Fail, cuEventRecord(s->agg[i].ready, s->compute));

  for (int lv = 0; lv < num_levels; ++lv) {
    struct level_state* lev = &s->levels[lv];

    aggregate_layout_destroy(&lev->agg_layout);

    uint64_t lod_tile_count[MAX_RANK];
    uint64_t lod_tiles_per_shard[MAX_RANK];
    for (int d = 0; d < rank; ++d) {
      lod_tile_count[d] =
        ceildiv(lev->dimensions[d].size, lev->dimensions[d].tile_size);
      uint64_t tps = lev->dimensions[d].tiles_per_shard;
      lod_tiles_per_shard[d] = (tps == 0) ? lod_tile_count[d] : tps;
    }

    CHECK(Fail,
          aggregate_layout_init(&lev->agg_layout, rank, lod_tile_count,
                                lod_tiles_per_shard, lev->layout.slot_count,
                                s->max_comp_chunk_bytes) == 0);

    size_t lev_comp_pool_bytes =
      lev->layout.slot_count * s->max_comp_chunk_bytes;
    for (int i = 0; i < 2; ++i) {
      CHECK(Fail,
            aggregate_slot_init(&lev->agg[i], &lev->agg_layout,
                                lev_comp_pool_bytes) == 0);
      CU(Fail, cuEventRecord(lev->agg[i].ready, s->compute));
    }
  }

  return 0;
Fail:
  return 1;
}

static int
seed_events(struct transpose_stream* s)
{
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventRecord(s->stage.slot[i].h_in.ready, s->compute));
    CU(Fail, cuEventRecord(s->stage.slot[i].t_h2d_start, s->compute));
    CU(Fail, cuEventRecord(s->stage.slot[i].t_scatter_start, s->compute));
    CU(Fail, cuEventRecord(s->stage.slot[i].t_scatter_end, s->compute));
  }
  CU(Fail, cuEventRecord(s->pool_A.ready, s->compute));
  CU(Fail, cuEventRecord(s->pool_B.ready, s->compute));
  CU(Fail, cuEventRecord(s->pool_A_host.ready, s->compute));
  CU(Fail, cuEventRecord(s->pool_B_host.ready, s->compute));
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventRecord(s->flush[i].t_compress_start, s->compute));
    CU(Fail, cuEventRecord(s->flush[i].t_d2h_start, s->compute));
    CU(Fail, cuEventRecord(s->flush[i].t_lod_start, s->compute));
    CU(Fail, cuEventRecord(s->flush[i].t_lod_end, s->compute));
    CU(Fail, cuEventRecord(s->flush[i].ready, s->compute));
  }

  return 0;
Fail:
  return 1;
}

int
transpose_stream_create(const struct transpose_stream_configuration* config,
                        struct transpose_stream* out)
{
  CHECK(Fail, config);
  CHECK(Fail, out);

  *out = (struct transpose_stream){
    .writer = { .append = transpose_stream_append,
                .flush = transpose_stream_flush },
    .config = *config,
  };

  CHECK(Fail, config->bytes_per_element > 0);
  CHECK(Fail, config->buffer_capacity_bytes > 0);
  out->config.buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;
  CHECK(Fail, config->rank > 0);
  CHECK(Fail, config->rank <= MAX_RANK / 2);
  CHECK(Fail, config->dimensions);

  CHECK(Fail, init_cuda_streams_and_events(out) == 0);
  CHECK(Fail, init_l0_layout(out) == 0);
  CHECK(Fail, init_staging_buffers(out) == 0);
  CHECK(Fail, init_lod_plan(out) == 0);
  CHECK(Fail, init_level_states(out) == 0);
  CHECK(Fail, init_tile_pools(out) == 0);
  CHECK(Fail, init_compression(out) == 0);
  CHECK(Fail, init_l0_aggregate_and_shards(out) == 0);
  CHECK(Fail, seed_events(out) == 0);

  CU(Fail, cuStreamSynchronize(out->compute));

  out->metrics = (struct stream_metrics){
    .memcpy   = { .name = "Memcpy",    .best_ms = 1e30f },
    .h2d      = { .name = "H2D",       .best_ms = 1e30f },
    .scatter  = { .name = "Scatter",   .best_ms = 1e30f },
    .lod      = { .name = "LOD",       .best_ms = 1e30f },
    .compress = { .name = "Compress",  .best_ms = 1e30f },
    .aggregate= { .name = "Aggregate", .best_ms = 1e30f },
    .d2h      = { .name = "D2H",       .best_ms = 1e30f },
  };

  return 0;

Fail:
  transpose_stream_destroy(out);
  return 1;
}

static struct writer_result
transpose_stream_append(struct writer* self, struct slice input)
{
  struct transpose_stream* s =
    container_of(self, struct transpose_stream, writer);
  const size_t bpe = s->config.bytes_per_element;
  const size_t buffer_capacity = s->config.buffer_capacity_bytes;
  const uint8_t* src = (const uint8_t*)input.beg;
  const uint8_t* end = (const uint8_t*)input.end;

  while (src < end) {
    const uint64_t epoch_remaining =
      s->layout.epoch_elements - (s->cursor % s->layout.epoch_elements);
    const uint64_t input_remaining = (uint64_t)(end - src) / bpe;
    const uint64_t elements_this_pass =
      epoch_remaining < input_remaining ? epoch_remaining : input_remaining;
    const uint64_t bytes_this_pass = elements_this_pass * bpe;

    {
      uint64_t written = 0;
      while (written < bytes_this_pass) {
        const size_t space = buffer_capacity - s->stage.fill;
        const uint64_t remaining = bytes_this_pass - written;
        const size_t payload = space < remaining ? space : (size_t)remaining;

        if (s->stage.fill == 0) {
          const int si = s->stage.current;
          struct staging_slot* ss = &s->stage.slot[si];
          CU(Error, cuEventSynchronize(ss->h_in.ready));

          if (s->cursor > 0) {
            accumulate_metric_cu(&s->metrics.h2d, ss->t_h2d_start, ss->h_in.ready);
            accumulate_metric_cu(&s->metrics.scatter, ss->t_scatter_start, ss->t_scatter_end);
          }
        }

        {
          struct platform_clock mc = { 0 };
          platform_toc(&mc);
          memcpy((uint8_t*)s->stage.slot[s->stage.current].h_in.data +
                   s->stage.fill,
                 src + written,
                 payload);
          accumulate_metric_ms(&s->metrics.memcpy,
                               (float)(platform_toc(&mc) * 1000.0));
        }
        s->stage.fill += payload;
        written += payload;

        if (s->stage.fill == buffer_capacity || written == bytes_this_pass) {
          if (dispatch_scatter(s))
            goto Error;
          s->stage.fill = 0;
        }
      }
    }
    src += bytes_this_pass;

    if (s->cursor % s->layout.epoch_elements == 0 && s->cursor > 0) {
      struct writer_result fr = flush_epoch(s);
      if (fr.error)
        return writer_error_at(src, end);
    }
  }

  return (struct writer_result){ .error = 0,
                                 .rest = { .beg = src, .end = end } };

Error:
  return writer_error_at(src, end);
}

static struct writer_result
transpose_stream_flush(struct writer* self)
{
  struct transpose_stream* s =
    container_of(self, struct transpose_stream, writer);

  if (s->stage.fill > 0) {
    if (dispatch_scatter(s))
      return writer_error();
    s->stage.fill = 0;
  }

  struct writer_result r = drain_pending_flush(s);
  if (r.error)
    return r;

  if (s->cursor % s->layout.epoch_elements != 0 || s->cursor == 0) {
    r = flush_epoch_sync(s);
    if (r.error)
      return r;
  }

  if (s->config.compress && s->config.shard_sink) {
    // Flush partial LOD epoch: if lod_epoch_counter > 0, fire with partial data
    if (s->num_levels > 0 && s->lod_epoch_counter > 0) {
      uint64_t actual_dim0 =
        s->lod_epoch_counter * s->config.dimensions[0].tile_size;

      int ffc = s->pool_current;
      CU(Error, cuEventRecord(s->flush[ffc].t_lod_start, s->compute));
      uint8_t lod_firing[MAX_LOD_LEVELS];
      int num_lod_fired = fire_lod(s, actual_dim0, lod_firing);
      if (num_lod_fired < 0)
        return writer_error();
      CU(Error, cuEventRecord(s->flush[ffc].t_lod_end, s->compute));
      s->flush[ffc].lod_fired = 1;

      s->lod_epoch_counter = 0;
    }

    // Emit partial shards
    for (int lv = 0; lv < s->num_levels; ++lv) {
      struct level_state* lev = &s->levels[lv];
      if (lev->shard.epoch_in_shard > 0) {
        if (emit_shards(&lev->shard))
          return writer_error();
      }
    }

    if (s->shard.epoch_in_shard > 0)
      return emit_shards(&s->shard) ? writer_error() : writer_ok();
    return writer_ok();
  }

  if (s->config.sink)
    return writer_flush(s->config.sink);

  return writer_ok();

Error:
  return writer_error();
}
