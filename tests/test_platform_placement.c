#define _GNU_SOURCE

#include "platform/placement.h"
#include "platform/platform.h"
#include "test_placement.h"
#include "util/prelude.h"

#include <stdlib.h>

static int
test_unavailable_placement(void)
{
  const struct test_placement_state before = test_placement_state();
  struct platform_placement_scope scope;
  CHECK(Fail, platform_pci_numa_node(NULL) == -1);
  CHECK(Fail, platform_pci_numa_node("no such PCI device") == -1);
  CHECK(Fail, platform_pci_numa_node("0000:ff:ff.9") == -1);
  CHECK(Fail, platform_placement_create(-1) == NULL);
  CHECK(Fail, platform_placement_create(1024) == NULL);
  platform_placement_enter(NULL, 1, &scope);
  CHECK(Fail, platform_placement_leave(&scope) == 0);
  CHECK(Fail, platform_placement_leave(&scope) == 0);
  CHECK(Fail, test_placement_unchanged(&before));
  return 0;
Fail:
  return 1;
}

#if defined(__linux__) && defined(SYS_getcpu)

static int
current_node(void)
{
  unsigned node = 0;
  return syscall(SYS_getcpu, NULL, &node, NULL) == 0 ? (int)node : -1;
}

static void
capture_worker(void* arg)
{
  *(struct test_placement_state*)arg = test_placement_state();
}

static int
test_restore_and_worker_inheritance(void)
{
  int failed = 1;
  struct platform_thread* worker = NULL;
  struct platform_placement_scope outer = { 0 }, inner = { 0 };
  const struct test_placement_state before = test_placement_state();
  struct platform_placement* placement =
    platform_placement_create(current_node());
  if (!placement) {
    log_info("no local CPU topology available");
    return 0;
  }
  platform_placement_enter(placement, 1, &outer);
  const struct test_placement_state during = test_placement_state();
  if (before.has_affinity && during.has_affinity) {
    cpu_set_t intersection;
    CPU_AND(&intersection, &before.cpus, &during.cpus);
    CHECK(Done, CPU_COUNT(&during.cpus) > 0);
    CHECK(Done, CPU_EQUAL(&intersection, &during.cpus));
  }
  if (outer.restore_memory_policy) {
    CHECK(Done, before.has_policy && before.mode == MPOL_DEFAULT);
    CHECK(Done, during.has_policy);
    CHECK(Done, during.mode == (MPOL_PREFERRED | MPOL_F_STATIC_NODES));
  }
  platform_placement_enter(placement, 1, &inner);
  CHECK(Done, platform_placement_leave(&inner) == 0);
  CHECK(Done, test_placement_unchanged(&during));
  struct test_placement_state inherited;
  worker = platform_thread_start(capture_worker, &inherited);
  CHECK(Done, worker);
  platform_placement_destroy(placement);
  placement = NULL;
  CHECK(Done, platform_placement_leave(&outer) == 0);
  const int joined = platform_thread_join(worker);
  worker = NULL;
  CHECK(Done, joined == 0);
  CHECK(Done, test_placement_equal(&during, &inherited));
  CHECK(Done, test_placement_unchanged(&before));
  failed = 0;
Done:
  platform_placement_leave(&inner);
  platform_placement_leave(&outer);
  if (worker)
    platform_thread_join(worker);
  platform_placement_destroy(placement);
  return failed;
}

static int
test_later_cpu_restriction(void)
{
  const struct test_placement_state before = test_placement_state();
  if (!before.has_affinity)
    return 0;
  int failed = 1;
  struct platform_placement_scope scope = { 0 };
  struct platform_placement* placement =
    platform_placement_create(current_node());
  if (!placement)
    return 0;
  cpu_set_t one;
  CPU_ZERO(&one);
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &before.cpus)) {
      CPU_SET(cpu, &one);
      break;
    }
  CHECK(Done, sched_setaffinity(0, sizeof(one), &one) == 0);
  const struct test_placement_state restricted = test_placement_state();
  platform_placement_enter(placement, 0, &scope);
  CHECK(Done, test_placement_unchanged(&restricted));
  CHECK(Done, platform_placement_leave(&scope) == 0);
  CHECK(Done, test_placement_unchanged(&restricted));
  failed = 0;
Done:
  platform_placement_leave(&scope);
  failed |= sched_setaffinity(0, sizeof(before.cpus), &before.cpus) != 0;
  platform_placement_destroy(placement);
  return failed;
}

#if defined(SYS_get_mempolicy) && defined(SYS_set_mempolicy)
static int
test_explicit_memory_policy(void)
{
  const struct test_placement_state before = test_placement_state();
  if (!before.has_policy) {
    log_info("memory-policy syscalls unavailable; fallback tested");
    return 0;
  }
  const int node = current_node();
  struct platform_placement* placement = platform_placement_create(node);
  if (!placement)
    return 0;
  unsigned long nodes[1024 / (8 * sizeof(unsigned long))] = { 0 };
  nodes[(unsigned)node / (8 * sizeof(unsigned long))] =
    1ul << ((unsigned)node % (8 * sizeof(unsigned long)));
  if (syscall(SYS_set_mempolicy, MPOL_BIND, nodes, 1024) != 0) {
    platform_placement_destroy(placement);
    log_info("cannot install a test policy; fallback tested");
    return 0;
  }
  const struct test_placement_state bound = test_placement_state();
  struct platform_placement_scope scope;
  platform_placement_enter(placement, 1, &scope);
  const struct test_placement_state during = test_placement_state();
  int failed = !during.has_policy || during.mode != bound.mode ||
               memcmp(during.nodes, bound.nodes, sizeof(during.nodes));
  failed |= platform_placement_leave(&scope);
  failed |= !test_placement_unchanged(&bound);
  failed |= syscall(SYS_set_mempolicy,
                    before.mode,
                    before.mode == MPOL_DEFAULT ? NULL : before.nodes,
                    before.mode == MPOL_DEFAULT ? 0 : 1024) != 0;
  failed |= !test_placement_unchanged(&before);
  platform_placement_destroy(placement);
  return failed != 0;
}
#endif
#endif

int
main(void)
{
  int failed = test_unavailable_placement();
#if defined(__linux__) && defined(SYS_getcpu)
  failed += test_restore_and_worker_inheritance();
  failed += test_later_cpu_restriction();
#if defined(SYS_get_mempolicy) && defined(SYS_set_mempolicy)
  failed += test_explicit_memory_policy();
#endif
#endif
  if (failed)
    log_error("%d placement tests FAILED", failed);
  else
    log_info("placement tests PASSED");
  return failed != 0;
}
