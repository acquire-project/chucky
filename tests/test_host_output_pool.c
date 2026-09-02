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
make_pool(void)
{
  const size_t page = platform_page_alignment();
  return host_output_pool_create(
    page, page, (struct host_output_allocator){ 0 });
}

static void
finish_output(struct host_output* output, size_t borrowed, int sealed)
{
  if (!output->group)
    return;
  if (!sealed)
    host_output_group_seal(output->group);
  while (borrowed-- > 0)
    host_output_group_complete(output->group);
  output->group = NULL;
}

static int
test_size_alignment(void)
{
  size_t bytes = 0;
  CHECK(Fail, host_output_size(4097, 4096, &bytes) == 0);
  CHECK(Fail, bytes == 8192);
  CHECK(Fail, host_output_size(0, 4096, &bytes) != 0);
  return 0;
Fail:
  return 1;
}

static int
test_groups_release_independently(void)
{
  struct host_output_pool* pool = make_pool();
  struct host_output first = { 0 };
  struct host_output second = { 0 };
  struct host_output next = { 0 };
  size_t first_borrowed = 0;
  size_t second_borrowed = 0;
  int first_sealed = 0;
  int second_sealed = 0;
  CHECK(Fail, pool);
  CHECK(Cleanup, host_output_pool_acquire(pool, &first) == 0);
  CHECK(Cleanup, host_output_pool_acquire(pool, &second) == 0);
  CHECK(Cleanup, host_output_group_retain(first.group) == 0);
  first_borrowed++;
  CHECK(Cleanup, host_output_group_retain(first.group) == 0);
  first_borrowed++;
  CHECK(Cleanup, host_output_group_retain(second.group) == 0);
  second_borrowed++;
  host_output_group_seal(first.group);
  first_sealed = 1;
  host_output_group_seal(second.group);
  second_sealed = 1;

  host_output_group_complete(second.group);
  second_borrowed--;
  second.group = NULL;
  CHECK(Cleanup, host_output_pool_acquire(pool, &next) == 0);
  CHECK(Cleanup, next.data == second.data);
  finish_output(&next, 0, 0);

  host_output_group_complete(first.group);
  first_borrowed--;
  host_output_group_complete(first.group);
  first_borrowed--;
  first.group = NULL;
  host_output_pool_destroy(pool);
  return 0;

Cleanup:
  finish_output(&first, first_borrowed, first_sealed);
  finish_output(&second, second_borrowed, second_sealed);
  finish_output(&next, 0, 0);
  host_output_pool_destroy(pool);
Fail:
  return 1;
}

static int
test_exhaustion_blocks(void)
{
  struct host_output_pool* pool = make_pool();
  struct host_output held[HOST_OUTPUT_COUNT] = { 0 };
  struct acquire_call call = { .pool = pool };
  test_thread* thread = NULL;
  CHECK(Fail, pool);
  for (size_t i = 0; i < HOST_OUTPUT_COUNT; ++i)
    CHECK(Cleanup, host_output_pool_acquire(pool, &held[i]) == 0);
  CHECK(Cleanup, test_thread_start(&thread, acquire_on_thread, &call) == 0);
  CHECK(Cleanup, test_wait_flag(&call.entered, 1000) == 0);
  CHECK(Cleanup, test_wait_flag(&call.done, 20) == -1);

  finish_output(&held[0], 0, 0);
  CHECK(Cleanup, test_wait_flag(&call.done, 1000) == 0);
  CHECK(Cleanup, call.result == 0);
  CHECK(Cleanup, test_thread_join(thread) == 0);
  thread = NULL;
  finish_output(&held[1], 0, 0);
  finish_output(&call.output, 0, 0);
  host_output_pool_destroy(pool);
  return 0;

Cleanup:
  for (size_t i = 0; i < HOST_OUTPUT_COUNT; ++i)
    finish_output(&held[i], 0, 0);
  if (thread)
    test_thread_join(thread);
  finish_output(&call.output, 0, 0);
  host_output_pool_destroy(pool);
Fail:
  return 1;
}

static int
test_destroy_wakes_waiter_and_waits_for_output(void)
{
  struct host_output_pool* pool = make_pool();
  struct host_output held[HOST_OUTPUT_COUNT] = { 0 };
  struct acquire_call acquire = { .pool = pool };
  struct destroy_call destroy = { .pool = pool };
  test_thread* acquire_thread = NULL;
  test_thread* destroy_thread = NULL;
  CHECK(Fail, pool);
  for (size_t i = 0; i < HOST_OUTPUT_COUNT; ++i)
    CHECK(Cleanup, host_output_pool_acquire(pool, &held[i]) == 0);
  CHECK(Cleanup,
        test_thread_start(&acquire_thread, acquire_on_thread, &acquire) == 0);
  CHECK(Cleanup, test_wait_flag(&acquire.entered, 1000) == 0);
  CHECK(Cleanup, test_wait_flag(&acquire.done, 20) == -1);
  CHECK(Cleanup,
        test_thread_start(&destroy_thread, destroy_on_thread, &destroy) == 0);
  CHECK(Cleanup, test_wait_flag(&destroy.entered, 1000) == 0);
  CHECK(Cleanup, test_wait_flag(&acquire.done, 1000) == 0);
  CHECK(Cleanup, acquire.result != 0);
  CHECK(Cleanup, test_wait_flag(&destroy.done, 20) == -1);

  for (size_t i = 0; i < HOST_OUTPUT_COUNT; ++i)
    finish_output(&held[i], 0, 0);
  CHECK(Cleanup, test_wait_flag(&destroy.done, 1000) == 0);
  CHECK(Cleanup, test_thread_join(acquire_thread) == 0);
  acquire_thread = NULL;
  CHECK(Cleanup, test_thread_join(destroy_thread) == 0);
  destroy_thread = NULL;
  return 0;

Cleanup:
  for (size_t i = 0; i < HOST_OUTPUT_COUNT; ++i)
    finish_output(&held[i], 0, 0);
  if (acquire_thread)
    test_thread_join(acquire_thread);
  finish_output(&acquire.output, 0, 0);
  if (destroy_thread)
    test_thread_join(destroy_thread);
  else
    host_output_pool_destroy(pool);
Fail:
  return 1;
}

int
main(void)
{
  return test_size_alignment() || test_groups_release_independently() ||
         test_exhaustion_blocks() ||
         test_destroy_wakes_waiter_and_waits_for_output();
}
