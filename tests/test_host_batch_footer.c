#include "platform/platform.h"
#include "stream/host_output_pool.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr/crc32c.h"
#include "zarr/io_scheduler.h"
#include "zarr/shard_delivery.h"
#include "zarr/shard_pool_fs.h"
#include "zarr/shard_write_plan.h"

#include <inttypes.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

struct footer_sink
{
  struct shard_sink base;
  struct shard_pool* pool;
  struct io_backend inner;
  const uint8_t* footer;
  struct host_output output[2];
  int borrowed;
  _Atomic int footer_entered;
  _Atomic int footer_waiting;
  _Atomic int release_footer;
  _Atomic int invalid_ownership;
};

static void
execute_write(void* ctx, const struct io_request* request)
{
  struct footer_sink* sink = (struct footer_sink*)ctx;
  if (request->op == IO_OP_WRITE) {
    if (request->offset == 0) {
      const int which = request->payload == sink->output[0].data ? 0 : 1;
      if (request->payload != sink->output[which].data || request->owned ||
          !request->finished ||
          request->finished_ctx != sink->output[which].group)
        atomic_store(&sink->invalid_ownership, 1);
    } else {
      if (request->finished || request->finished_ctx ||
          (sink->borrowed && (request->payload != sink->footer ||
                              request->owned || request->owned_free)) ||
          (!sink->borrowed && (!request->owned || !request->owned_free)))
        atomic_store(&sink->invalid_ownership, 1);
      if (sink->borrowed && !atomic_exchange(&sink->footer_entered, 1)) {
        while (!atomic_load(&sink->release_footer))
          platform_sleep_ns(1000000LL);
      }
    }
  }
  sink->inner.execute(sink->inner.ctx, request);
}

static struct io_backend
wrap_backend(void* ctx, struct io_backend inner)
{
  struct footer_sink* sink = (struct footer_sink*)ctx;
  sink->inner = inner;
  return (struct io_backend){ .ctx = sink, .execute = execute_write };
}

static struct shard_writer*
open_shard(struct shard_sink* self, uint8_t level, uint64_t shard)
{
  (void)level;
  struct footer_sink* sink = (struct footer_sink*)self;
  char key[64];
  snprintf(key, sizeof(key), "shard-%" PRIu64, shard);
  return sink->pool->open(sink->pool, 0, key);
}

static struct io_event
record_fence(struct shard_sink* self)
{
  struct footer_sink* sink = (struct footer_sink*)self;
  return sink->pool->record_fence(sink->pool);
}

static void
wait_fence(struct shard_sink* self, struct io_event event)
{
  struct footer_sink* sink = (struct footer_sink*)self;
  if (event.seq != 0)
    atomic_store(&sink->footer_waiting, 1);
  sink->pool->wait_fence(sink->pool, event);
}

struct delivery
{
  struct host_batch* batch;
  struct shard_state** shards;
  struct shard_sink* sink;
  _Atomic int done;
  int result;
};

static void
deliver_on_thread(void* arg)
{
  struct delivery* call = (struct delivery*)arg;
  call->result =
    deliver_host_batch(call->batch, call->shards, call->sink, NULL, NULL);
  atomic_store(&call->done, 1);
}

static int
check_shard(const char* root, int generation, size_t payload, uint8_t fill)
{
  char path[1024];
  snprintf(path, sizeof(path), "%s/shard-%d", root, generation);
  FILE* file = fopen(path, "rb");
  CHECK(Fail, file);
  uint8_t data[128];
  const size_t bytes = fread(data, 1, sizeof(data), file);
  const int read_failed = ferror(file);
  fclose(file);
  CHECK(Fail, !read_failed && bytes == payload + 20);
  for (size_t i = 0; i < payload; ++i)
    CHECK(Fail, data[i] == fill);
  uint64_t index[2];
  memcpy(index, data + payload, sizeof(index));
  CHECK(Fail, index[0] == 0 && index[1] == payload);
  uint32_t checksum;
  memcpy(&checksum, data + payload + sizeof(index), sizeof(checksum));
  CHECK(Fail, checksum == crc32c(data + payload, sizeof(index)));
  return 0;
Fail:
  return 1;
}

static int
test_footer_delivery(size_t alignment, int unaligned)
{
  const size_t page = 64;
  int result = 1;
  char root[512] = { 0 };
  uint8_t* allocation = NULL;
  struct host_output_pool* output_pool = NULL;
  struct host_output output[2] = { 0 };
  struct host_batch batch = { 0 };
  struct footer_sink sink = {
    .base = { .open = open_shard,
              .record_fence = record_fence,
              .wait_fence = wait_fence },
    .borrowed = alignment != 0 && !unaligned,
  };
  test_thread* thread = NULL;
  struct delivery call = { .batch = &batch, .sink = &sink.base };
  CHECK(Cleanup, test_tmpdir_create(root, sizeof(root)) == 0);
  allocation = (uint8_t*)platform_aligned_alloc(page, page * 2);
  CHECK(Cleanup, allocation);
  sink.footer = allocation + (unaligned ? 1 : 0);
  output_pool = host_output_pool_create(
    page * 2, page, (struct host_output_allocator){ 0 });
  CHECK(Cleanup, output_pool);
  for (int i = 0; i < 2; ++i) {
    CHECK(Cleanup, host_output_pool_acquire(output_pool, &output[i]) == 0);
    sink.output[i] = output[i];
    memset(output[i].data, 0x11 * (i + 1), output[i].capacity);
  }
  const struct io_scheduler_limits limits = { .workers = 2 };
  sink.pool = shard_pool_fs_create_wrapped(
    root,
    1,
    0,
    &limits,
    (struct shard_pool_fs_wrapper){ .ctx = &sink, .wrap = wrap_backend });
  CHECK(Cleanup, sink.pool);

  uint64_t index[2] = { UINT64_MAX, UINT64_MAX };
  struct active_shard active = {
    .index = index,
    .footer_buf = alignment ? (uint8_t*)sink.footer : NULL,
  };
  struct shard_state shard = {
    .shard_inner_count = 1,
    .chunks_per_shard_inner = 1,
    .chunks_per_shard_total = 1,
    .chunks_per_shard_append = 1,
    .shards = &active,
    .footer_capacity = page,
  };
  struct shard_state* shards[1] = { &shard };
  call.shards = shards;
  size_t offset = 0;
  size_t payload = 70;
  struct host_batch_run run = {
    .active_count = 1,
    .chunks_per_shard_inner = 1,
    .finalizes = 1,
    .ends_generation_run = 1,
    .data = output[0].data,
    .page_size = alignment,
    .payload_bytes = payload,
    .offsets = &offset,
    .chunk_sizes = &payload,
  };
  batch = (struct host_batch){
    .runs = &run,
    .run_count = 1,
    .nlod = 1,
    .storage = alignment ? HOST_BATCH_PAGE_PADDED : HOST_BATCH_PACKED,
    .shard_alignment = alignment,
    .output_group = output[0].group,
  };
  output[0].group = NULL;
  CHECK(Cleanup,
        deliver_host_batch(&batch, shards, &sink.base, NULL, NULL) == 0);
  CHECK(Cleanup, !batch.output_group && shard.shard_epoch == 1);
  if (sink.borrowed) {
    CHECK(Cleanup, test_wait_flag(&sink.footer_entered, 1000) == 0);
    CHECK(Cleanup, !atomic_load(&sink.invalid_ownership));
    CHECK(Cleanup, active.footer_io_done.seq != 0);
  }

  payload = 80;
  run.flat_shard = 1;
  run.data = output[1].data;
  run.payload_bytes = payload;
  batch.output_group = output[1].group;
  output[1].group = NULL;
  if (sink.borrowed) {
    CHECK(Cleanup, test_thread_start(&thread, deliver_on_thread, &call) == 0);
    CHECK(Cleanup, test_wait_flag(&sink.footer_waiting, 1000) == 0);
    CHECK(Cleanup, test_wait_flag(&call.done, 20) == -1);
    atomic_store(&sink.release_footer, 1);
    CHECK(Cleanup, test_wait_flag(&call.done, 1000) == 0);
    CHECK(Cleanup, test_thread_join(thread) == 0);
    thread = NULL;
    CHECK(Cleanup, call.result == 0);
  } else {
    CHECK(Cleanup,
          deliver_host_batch(&batch, shards, &sink.base, NULL, NULL) == 0);
    CHECK(Cleanup, active.footer_io_done.seq == 0);
  }
  CHECK(Cleanup, !batch.output_group && shard.shard_epoch == 2);
  CHECK(Cleanup, sink.pool->flush(sink.pool) == 0);
  CHECK(Cleanup, !atomic_load(&sink.invalid_ownership));
  CHECK(Cleanup, check_shard(root, 0, 70, 0x11) == 0);
  CHECK(Cleanup, check_shard(root, 1, 80, 0x22) == 0);
  result = 0;

Cleanup:
  atomic_store(&sink.release_footer, 1);
  if (thread)
    test_thread_join(thread);
  if (batch.output_group)
    host_output_group_seal(batch.output_group);
  for (int i = 0; i < 2; ++i)
    if (output[i].group)
      host_output_group_seal(output[i].group);
  shard_pool_destroy(sink.pool);
  host_output_pool_destroy(output_pool);
  platform_aligned_free(allocation);
  if (root[0])
    test_tmpdir_remove(root);
  return result;
}

int
main(void)
{
  return test_footer_delivery(64, 0) || test_footer_delivery(64, 1) ||
         test_footer_delivery(0, 0);
}
