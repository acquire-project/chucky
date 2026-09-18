#define _GNU_SOURCE

#include "platform/placement.h"

#include <stdlib.h>

#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sched.h>
#include <stdio.h>
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

int
platform_pci_numa_node(const char* pci_bus_id)
{
  unsigned domain, bus, device, function;
  char extra;
  if (!pci_bus_id ||
      sscanf(pci_bus_id,
             "%x:%x:%x.%x%c",
             &domain,
             &bus,
             &device,
             &function,
             &extra) != 4 ||
      bus > 255 || device > 31 || function > 7)
    return -1;
  char path[128];
  snprintf(path,
           sizeof(path),
           "/sys/bus/pci/devices/%04x:%02x:%02x.%x/numa_node",
           domain,
           bus,
           device,
           function);
  FILE* file = fopen(path, "r");
  if (!file)
    return -1;
  int node = -1;
  if (fscanf(file, "%d", &node) != 1)
    node = -1;
  fclose(file);
  return node >= 0 && node < (int)NODE_COUNT ? node : -1;
}

struct platform_placement*
platform_placement_create(int node)
{
  if (node < 0 || node >= (int)NODE_COUNT)
    return NULL;
  char path[128];
  snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
  FILE* file = fopen(path, "r");
  if (!file)
    return NULL;
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  int valid = 0;
  for (;;) {
    unsigned first, last;
    if (fscanf(file, "%u", &first) != 1)
      break;
    last = first;
    int next = fgetc(file);
    if (next == '-') {
      if (fscanf(file, "%u", &last) != 1)
        break;
      next = fgetc(file);
    }
    if (last < first || last >= CPU_SETSIZE)
      break;
    for (unsigned cpu = first; cpu <= last; ++cpu)
      CPU_SET(cpu, &cpus);
    if (next == '\n' || next == EOF) {
      valid = 1;
      break;
    }
    if (next != ',')
      break;
  }
  fclose(file);
  if (!valid)
    return NULL;

  struct platform_placement* placement = malloc(sizeof(*placement));
  if (placement) {
    placement->cpus = cpus;
    placement->node = node;
  }
  return placement;
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

int
platform_pci_numa_node(const char* pci_bus_id)
{
  (void)pci_bus_id;
  return -1;
}

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
