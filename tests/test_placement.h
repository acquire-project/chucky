#pragma once

#include <string.h>

#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

struct test_placement_state
{
#if defined(__linux__)
  cpu_set_t cpus;
  unsigned long nodes[1024 / (8 * sizeof(unsigned long))];
#endif
  int mode;
  int has_affinity;
  int has_policy;
};

static inline struct test_placement_state
test_placement_state(void)
{
  struct test_placement_state state = { 0 };
#if defined(__linux__)
  state.has_affinity =
    sched_getaffinity(0, sizeof(state.cpus), &state.cpus) == 0;
#if defined(SYS_get_mempolicy)
  state.has_policy =
    syscall(SYS_get_mempolicy, &state.mode, state.nodes, 1024, NULL, 0) == 0;
#endif
#endif
  return state;
}

static inline int
test_placement_equal(const struct test_placement_state* a,
                     const struct test_placement_state* b)
{
  if (a->has_affinity != b->has_affinity || a->has_policy != b->has_policy)
    return 0;
#if defined(__linux__)
  if (a->has_affinity && !CPU_EQUAL(&a->cpus, &b->cpus))
    return 0;
  if (a->has_policy &&
      (a->mode != b->mode || memcmp(a->nodes, b->nodes, sizeof(a->nodes))))
    return 0;
#endif
  return 1;
}

static inline int
test_placement_unchanged(const struct test_placement_state* before)
{
  const struct test_placement_state after = test_placement_state();
  return test_placement_equal(before, &after);
}
