#include "test_data.h"
#include "log/log.h"
#include "platform/platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- PRNG ---

static uint64_t
splitmix64(uint64_t* state)
{
  uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// --- Fill functions ---

// --- Rand pattern (pre-populated, uniform random 12-bit) ---

static uint16_t* rand_pattern_buf = NULL;
static size_t rand_pattern_len = 0;

void
rand_pattern_init(const struct dimension* dims, uint8_t rank, size_t nframes)
{
  size_t frame = 1;
  for (uint8_t i = 1; i < rank; ++i)
    frame *= dims[i].size;
  rand_pattern_len = nframes * frame;
  free(rand_pattern_buf);
  rand_pattern_buf = (uint16_t*)malloc(rand_pattern_len * sizeof(uint16_t));

  uint64_t rng = 0xdeadbeefcafebabeULL;
  for (size_t i = 0; i < rand_pattern_len; ++i)
    rand_pattern_buf[i] = (uint16_t)(splitmix64(&rng) & 0xFFF);
}

void
rand_pattern_free(void)
{
  free(rand_pattern_buf);
  rand_pattern_buf = NULL;
  rand_pattern_len = 0;
}

void
fill_rand(uint16_t* buf, size_t count, size_t offset, size_t total)
{
  (void)total;
  for (size_t done = 0; done < count;) {
    size_t src_off = (offset + done) % rand_pattern_len;
    size_t chunk = rand_pattern_len - src_off;
    if (chunk > count - done)
      chunk = count - done;
    memcpy(buf + done, rand_pattern_buf + src_off, chunk * sizeof(uint16_t));
    done += chunk;
  }
}

void
fill_zeros(uint16_t* buf, size_t count, size_t offset, size_t total)
{
  (void)offset;
  (void)total;
  memset(buf, 0, count * sizeof(uint16_t));
}

// --- XOR pattern ---

static uint16_t* xor_pattern_buf = NULL;
static size_t xor_pattern_len = 0;

void
xor_pattern_init(const struct dimension* dims, uint8_t rank, size_t nframes)
{
  size_t frame = 1;
  for (uint8_t i = 1; i < rank; ++i)
    frame *= dims[i].size;
  xor_pattern_len = nframes * frame;
  free(xor_pattern_buf);
  xor_pattern_buf = (uint16_t*)malloc(xor_pattern_len * sizeof(uint16_t));

  size_t strides[16];
  strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * dims[i + 1].size;

  for (size_t gi = 0; gi < xor_pattern_len; ++gi) {
    uint16_t v = 0;
    size_t rem = gi;
    for (uint8_t d = 0; d < rank; ++d) {
      size_t coord = rem / strides[d];
      rem %= strides[d];
      v ^= (uint16_t)coord;
    }
    xor_pattern_buf[gi] = v;
  }
}

void
xor_pattern_free(void)
{
  free(xor_pattern_buf);
  xor_pattern_buf = NULL;
  xor_pattern_len = 0;
}

void
fill_xor(uint16_t* buf, size_t count, size_t offset, size_t total)
{
  (void)total;
  for (size_t done = 0; done < count;) {
    size_t src_off = (offset + done) % xor_pattern_len;
    size_t piece = xor_pattern_len - src_off;
    if (piece > count - done)
      piece = count - done;
    memcpy(buf + done, xor_pattern_buf + src_off, piece * sizeof(uint16_t));
    done += piece;
  }
}

// Period of the active fill pattern, in elements (0 if the fill has none, e.g.
// fill_zeros). Block cycling staggers offsets across this so the pre-generated
// blocks are distinct instead of aliasing back to identical content.
size_t
fill_pattern_period(fill_fn fill)
{
  if (fill == fill_xor)
    return xor_pattern_len;
  if (fill == fill_rand)
    return rand_pattern_len;
  return 0;
}

// --- Helpers ---

size_t
dim_total_elements(const struct dimension* dims, uint8_t rank)
{
  size_t n = 1;
  for (uint8_t i = 0; i < rank; ++i)
    n *= dims[i].size;
  return n;
}

int
pump_data_bpe(struct writer* w, size_t total_elements, fill_fn fill, size_t bpe)
{
  return pump_data_modal(
    w, total_elements, fill, bpe, PUMP_BUSY_PRODUCER, NULL);
}

int
pump_data(struct writer* w, size_t total_elements, fill_fn fill)
{
  return pump_data_bpe(w, total_elements, fill, sizeof(uint16_t));
}

int
pump_blocks_alloc(struct pump_blocks* out,
                  size_t total_elements,
                  fill_fn fill,
                  size_t bpe,
                  enum pump_mode mode,
                  int measure_fill)
{
  const size_t nelements = PUMP_BLOCK_ELEMENTS;
  const size_t block_alloc = nelements * (bpe > 2 ? bpe : 2);

  out->block = NULL;
  out->count = 0;
  out->fill_s = 0.0;

  size_t nblocks = 1;
  if (mode == PUMP_CYCLE_BLOCKS) {
    size_t nappends = total_elements / nelements;
    if (total_elements % nelements)
      nappends += 1;
    if (nappends == 0)
      nappends = 1;

    nblocks = PUMP_CYCLE_BLOCK_COUNT;
    if (nblocks > nappends)
      nblocks = nappends;
    // Cycling N blocks costs N * block_alloc of host RAM (vs one buffer in the
    // other modes). Bound the total so large bytes-per-element runs don't blow
    // up the harness footprint; u16 (64 MiB/block) is unaffected.
    size_t max_by_mem = ((size_t)1 << 30) / block_alloc;
    if (max_by_mem < 1)
      max_by_mem = 1;
    if (nblocks > max_by_mem)
      nblocks = max_by_mem;
  }

  uint16_t** blocks = (uint16_t**)calloc(nblocks, sizeof(uint16_t*));
  if (!blocks)
    return 1;
  for (size_t b = 0; b < nblocks; ++b) {
    // calloc: for bpe>2 the fill writes only the low uint16_t lanes, so zero
    // the rest rather than ship uninitialized bytes to the writer.
    blocks[b] = (uint16_t*)calloc(1, block_alloc);
    if (!blocks[b]) {
      for (size_t j = 0; j < b; ++j)
        free(blocks[j]);
      free(blocks);
      return 1;
    }
  }
  out->block = blocks;
  out->count = nblocks;

  struct platform_clock fill_clock = { 0 };
  if (measure_fill)
    platform_toc(&fill_clock);
  if (mode == PUMP_CYCLE_BLOCKS) {
    // Stagger each block's start across the fill pattern's full period so the
    // blocks are genuinely distinct. A naive b*nelements offset aliases to
    // identical content whenever nelements is a multiple of the period.
    const size_t period = fill_pattern_period(fill);
    const size_t stride = (period && period >= nblocks) ? period / nblocks : 1;
    for (size_t b = 0; b < nblocks; ++b)
      fill(blocks[b], nelements, b * stride, total_elements);
  } else if (mode == PUMP_SINGLE_BLOCK) {
    fill(blocks[0],
         nelements < total_elements ? nelements : total_elements,
         0,
         total_elements);
  }
  if (measure_fill)
    out->fill_s = (double)platform_toc(&fill_clock);

  return 0;
}

void
pump_blocks_free(struct pump_blocks* b)
{
  for (size_t i = 0; i < b->count; ++i)
    free(b->block[i]);
  free(b->block);
  b->block = NULL;
  b->count = 0;
}

// Cycle through pre-generated distinct blocks (PUMP_CYCLE_BLOCKS), one per
// append with no in-loop fill, so per-append cost is just the writer.
static int
pump_cycle_blocks(struct writer* w,
                  size_t total_elements,
                  fill_fn fill,
                  size_t bpe,
                  double* out_fill_s)
{
  const size_t nelements = PUMP_BLOCK_ELEMENTS;
  struct pump_blocks blocks;
  if (pump_blocks_alloc(
        &blocks, total_elements, fill, bpe, PUMP_CYCLE_BLOCKS, out_fill_s != 0))
    return 1;
  if (out_fill_s)
    *out_fill_s = blocks.fill_s;

  int err = 0;
  size_t b = 0;
  for (size_t offset = 0; offset < total_elements; offset += nelements) {
    size_t n = nelements;
    if (offset + n > total_elements)
      n = total_elements - offset;
    uint16_t* data = blocks.block[b];
    b = (b + 1 == blocks.count) ? 0 : b + 1;
    struct slice input = { .beg = data, .end = (char*)data + n * bpe };
    struct writer_result r = writer_append_wait(w, input);
    if (r.error) {
      log_error("  append failed at offset %zu", offset);
      err = 1;
      break;
    }
  }

  if (!err) {
    struct writer_result r = writer_flush(w);
    err = r.error;
  }
  pump_blocks_free(&blocks);
  return err;
}

int
pump_data_modal(struct writer* w,
                size_t total_elements,
                fill_fn fill,
                size_t bpe,
                enum pump_mode mode,
                double* out_fill_s)
{
  if (out_fill_s)
    *out_fill_s = 0.0;

  if (mode == PUMP_CYCLE_BLOCKS)
    return pump_cycle_blocks(w, total_elements, fill, bpe, out_fill_s);

  const size_t nelements = PUMP_BLOCK_ELEMENTS;
  // Allocate max(n*bpe, n*2) so fill (which writes uint16_t) always fits.
  // calloc: for bpe>2 the fill writes only the low uint16_t lanes, so zero
  // the rest rather than ship uninitialized bytes to the writer.
  size_t alloc = nelements * (bpe > 2 ? bpe : 2);
  uint16_t* data = (uint16_t*)calloc(1, alloc);
  if (!data)
    return 1;

  struct platform_clock fill_clock = { 0 };
  double fill_s = 0.0;
  if (mode == PUMP_SINGLE_BLOCK) {
    if (out_fill_s)
      platform_toc(&fill_clock);
    fill(data,
         nelements < total_elements ? nelements : total_elements,
         0,
         total_elements);
    if (out_fill_s)
      fill_s += (double)platform_toc(&fill_clock);
  }

  for (size_t offset = 0; offset < total_elements; offset += nelements) {
    size_t n = nelements;
    if (offset + n > total_elements)
      n = total_elements - offset;
    if (mode == PUMP_BUSY_PRODUCER) {
      if (out_fill_s)
        platform_toc(&fill_clock);
      fill(data, n, offset, total_elements);
      if (out_fill_s)
        fill_s += (double)platform_toc(&fill_clock);
    }
    struct slice input = { .beg = data, .end = (char*)data + n * bpe };
    struct writer_result r = writer_append_wait(w, input);
    if (r.error) {
      log_error("  append failed at offset %zu", offset);
      free(data);
      return 1;
    }
  }

  struct writer_result r = writer_flush(w);
  free(data);
  if (out_fill_s)
    *out_fill_s = fill_s;
  return r.error;
}
