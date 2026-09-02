#include "platform/platform.h"
#include "stream/host_output_pool.h"
#include "test_platform.h"
#include "util/prelude.h"

#include <stdatomic.h>

struct acquire_call
{
  struct host_output_pool* pool;
  struct host_output output;
  _Atomic int entered;
  _Atomic int done;
  int result;
};

struct destroy_call
{
  struct host_output_pool* pool;
  _Atomic int entered;
  _Atomic int done;
};

static void
acquire_on_thread(void* arg)
{
  struct acquire_call* call = (struct acquire_call*)arg;
  atomic_store(&call->entered, 1);
  call->result = host_output_pool_acquire(call->pool, &call->output);
  atomic_store(&call->done, 1);
}

static void
destroy_on_thread(void* arg)
{
  struct destroy_call* call = (struct destroy_call*)arg;
  atomic_store(&call->entered, 1);
  host_output_pool_destroy(call->pool);
  atomic_store(&call->done, 1);
}

static struct host_output_pool*
make_pool(uint64_t count)
{
  const size_t page = platform_page_alignment();
  return host_output_pool_create(
    count, page, page, (struct host_output_allocator){ 0 });
}

static int
test_budget_resolution(void)
{
  size_t bytes = 0;
  uint64_t count = 0;
  CHECK(Fail, host_output_size(4097, 4096, &bytes) == 0);
  CHECK(Fail, bytes == 8192);
  CHECK(Fail, host_output_count(0, bytes, &count) == 0 && count == 2);
  CHECK(Fail, host_output_count(bytes, bytes, &count) == 0 && count == 1);
  CHECK(Fail,
        host_output_count(3 * bytes + bytes / 2, bytes, &count) == 0 &&
          count == 3);
  CHECK(Fail, host_output_count(bytes - 1, bytes, &count) != 0);
  return 0;

Fail:
  return 1;
}

static int
test_groups_release_independently(void)
{
  struct host_output_pool* pool = make_pool(2);
  CHECK(Fail, pool);
  struct host_output first = { 0 };
  struct host_output second = { 0 };
  struct host_output third = { 0 };
  uint64_t first_pending = 0;
  uint64_t second_pending = 0;
  int first_sealed = 0;
  int second_sealed = 0;
  CHECK(Cleanup, host_output_pool_acquire(pool, &first) == 0);
  CHECK(Cleanup, host_output_pool_acquire(pool, &second) == 0);
  CHECK(Cleanup, host_output_group_retain(first.group) == 0);
  first_pending++;
  CHECK(Cleanup, host_output_group_retain(first.group) == 0);
  first_pending++;
  CHECK(Cleanup, host_output_group_retain(second.group) == 0);
  second_pending++;
  host_output_group_seal(first.group);
  first_sealed = 1;
  host_output_group_seal(second.group);
  second_sealed = 1;

  host_output_group_complete(second.group);
  second_pending--;
  second.group = NULL;
  CHECK(Cleanup, host_output_pool_acquire(pool, &third) == 0);
  CHECK(Cleanup, third.data == second.data);
  host_output_group_seal(third.group);
  third.group = NULL;

  struct host_output_pool_stats stats;
  host_output_pool_get_stats(pool, &stats);
  CHECK(Cleanup, stats.buffers_in_use == 1);
  host_output_group_complete(first.group);
  first_pending--;
  host_output_pool_get_stats(pool, &stats);
  CHECK(Cleanup, stats.buffers_in_use == 1);
  host_output_group_complete(first.group);
  first_pending--;
  first.group = NULL;
  host_output_pool_get_stats(pool, &stats);
  CHECK(Cleanup, stats.buffers_in_use == 0);
  CHECK(Cleanup, stats.buffers_in_use_peak == 2);
  CHECK(Cleanup, stats.lifetime_count == 3);
  host_output_pool_destroy(pool);
  return 0;

Cleanup:
  host_output_pool_close(pool);
  while (first_pending > 0) {
    host_output_group_complete(first.group);
    first_pending--;
  }
  if (first.group && !first_sealed)
    host_output_group_seal(first.group);
  while (second_pending > 0) {
    host_output_group_complete(second.group);
    second_pending--;
  }
  if (second.group && !second_sealed)
    host_output_group_seal(second.group);
  if (third.group)
    host_output_group_seal(third.group);
  host_output_pool_destroy(pool);
Fail:
  return 1;
}

static int
test_exhaustion_blocks(void)
{
  struct host_output_pool* pool = make_pool(1);
  CHECK(Fail, pool);
  struct host_output held = { 0 };
  CHECK(Cleanup, host_output_pool_acquire(pool, &held) == 0);
  struct acquire_call call = { .pool = pool };
  test_thread* thread = NULL;
  CHECK(Cleanup, test_thread_start(&thread, acquire_on_thread, &call) == 0);
  CHECK(Cleanup, test_wait_flag(&call.entered, 1000) == 0);
  CHECK(Cleanup, test_wait_flag(&call.done, 20) == -1);

  host_output_group_seal(held.group);
  held.group = NULL;
  CHECK(Cleanup, test_wait_flag(&call.done, 1000) == 0);
  CHECK(Cleanup, call.result == 0);
  CHECK(Cleanup, test_thread_join(thread) == 0);
  thread = NULL;
  host_output_group_seal(call.output.group);
  call.output.group = NULL;

  struct host_output_pool_stats stats;
  host_output_pool_get_stats(pool, &stats);
  CHECK(Cleanup, stats.wait_calls == 2);
  CHECK(Cleanup, stats.wait_count == 2);
  CHECK(Cleanup, stats.wait_ms_max > 0.0);
  host_output_pool_destroy(pool);
  return 0;

Cleanup:
  host_output_pool_close(pool);
  if (held.group)
    host_output_group_seal(held.group);
  if (thread)
    test_thread_join(thread);
  if (call.output.group)
    host_output_group_seal(call.output.group);
  host_output_pool_destroy(pool);
Fail:
  return 1;
}

static int
test_close_wakes_waiter(void)
{
  struct host_output_pool* pool = make_pool(1);
  CHECK(Fail, pool);
  struct host_output held = { 0 };
  CHECK(Cleanup, host_output_pool_acquire(pool, &held) == 0);
  struct acquire_call call = { .pool = pool };
  test_thread* thread = NULL;
  CHECK(Cleanup, test_thread_start(&thread, acquire_on_thread, &call) == 0);
  CHECK(Cleanup, test_wait_flag(&call.entered, 1000) == 0);
  CHECK(Cleanup, test_wait_flag(&call.done, 20) == -1);
  host_output_pool_close(pool);
  CHECK(Cleanup, test_wait_flag(&call.done, 1000) == 0);
  CHECK(Cleanup, call.result != 0);
  CHECK(Cleanup, test_thread_join(thread) == 0);
  thread = NULL;
  host_output_group_seal(held.group);
  held.group = NULL;
  host_output_pool_destroy(pool);
  return 0;

Cleanup:
  host_output_pool_close(pool);
  if (thread)
    test_thread_join(thread);
  if (held.group)
    host_output_group_seal(held.group);
  if (call.output.group)
    host_output_group_seal(call.output.group);
  host_output_pool_destroy(pool);
Fail:
  return 1;
}

static int
test_close_rejects_free_buffer(void)
{
  struct host_output_pool* pool = make_pool(1);
  CHECK(Fail, pool);
  host_output_pool_close(pool);
  struct host_output output;
  CHECK(Cleanup, host_output_pool_acquire(pool, &output) != 0);
  host_output_pool_destroy(pool);
  return 0;

Cleanup:
  host_output_pool_destroy(pool);
Fail:
  return 1;
}

static int
test_destroy_wakes_waiter_and_waits_for_output(void)
{
  struct host_output_pool* pool = make_pool(1);
  struct host_output held = { 0 };
  struct acquire_call acquire = { .pool = pool };
  struct destroy_call destroy = { .pool = pool };
  test_thread* acquire_thread = NULL;
  test_thread* destroy_thread = NULL;
  CHECK(Fail, pool);
  CHECK(Cleanup, host_output_pool_acquire(pool, &held) == 0);

  CHECK(Cleanup,
        test_thread_start(&acquire_thread, acquire_on_thread, &acquire) == 0);
  CHECK(Cleanup, test_wait_flag(&acquire.entered, 1000) == 0);
  CHECK(Cleanup, test_wait_flag(&acquire.done, 20) == -1);

  CHECK(Cleanup,
        test_thread_start(&destroy_thread, destroy_on_thread, &destroy) == 0);
  CHECK(Cleanup, test_wait_flag(&destroy.entered, 1000) == 0);
  CHECK(Cleanup, test_wait_flag(&acquire.done, 1000) == 0);
  CHECK(Cleanup, acquire.result != 0);
  CHECK(Cleanup, test_thread_join(acquire_thread) == 0);
  acquire_thread = NULL;
  CHECK(Cleanup, test_wait_flag(&destroy.done, 20) == -1);

  host_output_group_seal(held.group);
  held.group = NULL;
  CHECK(Cleanup, test_wait_flag(&destroy.done, 1000) == 0);
  CHECK(Cleanup, test_thread_join(destroy_thread) == 0);
  destroy_thread = NULL;
  pool = NULL;
  return 0;

Cleanup:
  if (!destroy_thread)
    host_output_pool_close(pool);
  if (acquire_thread)
    test_thread_join(acquire_thread);
  if (held.group)
    host_output_group_seal(held.group);
  if (acquire.output.group)
    host_output_group_seal(acquire.output.group);
  if (destroy_thread) {
    test_thread_join(destroy_thread);
    pool = NULL;
  }
  host_output_pool_destroy(pool);
Fail:
  return 1;
}

int
main(void)
{
  if (test_budget_resolution())
    return 1;
  if (test_groups_release_independently())
    return 1;
  if (test_exhaustion_blocks())
    return 1;
  if (test_close_wakes_waiter())
    return 1;
  if (test_close_rejects_free_buffer())
    return 1;
  if (test_destroy_wakes_waiter_and_waits_for_output())
    return 1;
  return 0;
}
