#include "test_data.h"

#include "prelude.h"

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
  rand_pattern_buf =
    (uint16_t*)malloc(rand_pattern_len * sizeof(uint16_t));

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
pump_data_bpe(struct writer* w,
              size_t total_elements,
              fill_fn fill,
              size_t bpe)
{
  const size_t nelements = 32 * 1024 * 1024; // 32M elements
  // Allocate max(n*bpe, n*2) so fill (which writes uint16_t) always fits
  size_t alloc = nelements * (bpe > 2 ? bpe : 2);
  uint16_t* data = (uint16_t*)malloc(alloc);
  if (!data)
    return 1;

  for (size_t offset = 0; offset < total_elements; offset += nelements) {
    size_t n = nelements;
    if (offset + n > total_elements)
      n = total_elements - offset;
    fill(data, n, offset, total_elements);
    struct slice input = { .beg = data,
                           .end = (char*)data + n * bpe };
    struct writer_result r = writer_append_wait(w, input);
    if (r.error) {
      log_error("  append failed at offset %zu", offset);
      free(data);
      return 1;
    }
  }

  struct writer_result r = writer_flush(w);
  free(data);
  return r.error;
}

int
pump_data(struct writer* w, size_t total_elements, fill_fn fill)
{
  return pump_data_bpe(w, total_elements, fill, sizeof(uint16_t));
}
