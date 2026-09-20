#define _GNU_SOURCE

#include "platform/placement.h"
#include "log/log.h"
#include "platform/topology.h"

#include <stdlib.h>

#if defined(__linux__)
#include <errno.h>
#include <linux/mempolicy.h>
#include <sched.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#define NODE_COUNT 1024u
#define WORD_BITS (8u * sizeof(unsigned long))

struct platform_placement
{
  cpu_set_t cpus;
  int node;
};

struct platform_placement*
platform_placement_create(int node)
{
  if (node < 0)
    return NULL;
  if (node >= (int)NODE_COUNT) {
    log_warn("NUMA placement disabled: node %d exceeds the %u-node mask limit",
             node,
             NODE_COUNT);
    return NULL;
  }
  unsigned char members[CPU_SETSIZE];
  if (platform_numa_cpus(node, members, sizeof(members)))
    return NULL;
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (members[cpu])
      CPU_SET(cpu, &cpus);

  struct platform_placement* placement = malloc(sizeof(*placement));
  if (placement) {
    placement->cpus = cpus;
    placement->node = node;
  }
  return placement;
}

static void
warn_affinity_size(void)
{
  // Append also reads affinity; do not repeat this warning on every call.
  static atomic_flag warned = ATOMIC_FLAG_INIT;
  if (!atomic_flag_test_and_set_explicit(&warned, memory_order_relaxed))
    log_warn("NUMA CPU placement disabled: the kernel affinity mask exceeds "
             "the %d-CPU mask limit",
             CPU_SETSIZE);
}

void
platform_placement_enter(const struct platform_placement* placement,
                         int place_memory,
                         struct platform_placement_scope* scope)
{
  *scope = (struct platform_placement_scope){ 0 };
  if (!placement)
    return;

  cpu_set_t before, local;
  if (sched_getaffinity(0, sizeof(before), &before) == 0) {
    CPU_AND(&local, &before, &placement->cpus);
    if (CPU_COUNT(&local) && !CPU_EQUAL(&before, &local) &&
        sizeof(before) <= sizeof(scope->saved_affinity) &&
        sched_setaffinity(0, sizeof(local), &local) == 0) {
      memcpy(scope->saved_affinity, &before, sizeof(before));
      scope->restore_affinity = 1;
    }
  } else if (errno == EINVAL) {
    warn_affinity_size();
  }

#if defined(SYS_get_mempolicy) && defined(SYS_set_mempolicy)
  int mode = -1;
  unsigned long nodes[NODE_COUNT / WORD_BITS] = { 0 };
  if (!place_memory ||
      syscall(SYS_get_mempolicy, &mode, NULL, 0, NULL, 0) != 0 ||
      mode != MPOL_DEFAULT ||
      syscall(SYS_get_mempolicy,
              NULL,
              nodes,
              NODE_COUNT,
              NULL,
              MPOL_F_MEMS_ALLOWED) != 0)
    return;
  const unsigned node = (unsigned)placement->node;
  if (!(nodes[node / WORD_BITS] & (1ul << (node % WORD_BITS))))
    return;
  memset(nodes, 0, sizeof(nodes));
  nodes[node / WORD_BITS] = 1ul << (node % WORD_BITS);
  scope->restore_memory_policy = syscall(SYS_set_mempolicy,
                                         MPOL_PREFERRED | MPOL_F_STATIC_NODES,
                                         nodes,
                                         NODE_COUNT) == 0;
#else
  (void)place_memory;
#endif
}

int
platform_placement_leave(struct platform_placement_scope* scope)
{
  int failed = 0;
#if defined(SYS_set_mempolicy)
  if (scope->restore_memory_policy)
    failed |= syscall(SYS_set_mempolicy, MPOL_DEFAULT, NULL, 0) != 0;
#endif
  if (scope->restore_affinity) {
    cpu_set_t before;
    memcpy(&before, scope->saved_affinity, sizeof(before));
    failed |= sched_setaffinity(0, sizeof(before), &before) != 0;
  }
  *scope = (struct platform_placement_scope){ 0 };
  return failed;
}

#else

struct platform_placement*
platform_placement_create(int node)
{
  (void)node;
  return NULL;
}

void
platform_placement_enter(const struct platform_placement* placement,
                         int place_memory,
                         struct platform_placement_scope* scope)
{
  (void)placement;
  (void)place_memory;
  *scope = (struct platform_placement_scope){ 0 };
}

int
platform_placement_leave(struct platform_placement_scope* scope)
{
  *scope = (struct platform_placement_scope){ 0 };
  return 0;
}

#endif

void
platform_placement_destroy(struct platform_placement* placement)
{
  free(placement);
}
