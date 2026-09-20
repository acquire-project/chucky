#include "platform/topology.h"

#include "log/log.h"

#include <errno.h>
#include <limits.h>
#include <string.h>

static enum platform_cpu_list_result
read_cpu(FILE* file, unsigned* cpu, int* next)
{
  int c = fgetc(file);
  if (c < '0' || c > '9')
    return PLATFORM_CPU_LIST_INVALID;
  *cpu = 0;
  do {
    const unsigned digit = (unsigned)(c - '0');
    if (*cpu > (UINT_MAX - digit) / 10)
      return PLATFORM_CPU_LIST_TOO_LARGE;
    *cpu = *cpu * 10 + digit;
    c = fgetc(file);
  } while (c >= '0' && c <= '9');
  *next = c;
  return PLATFORM_CPU_LIST_OK;
}

enum platform_cpu_list_result
platform_parse_cpu_list(FILE* file, unsigned char* cpus, size_t count)
{
  if (!file || !cpus || !count)
    return PLATFORM_CPU_LIST_INVALID;
  memset(cpus, 0, count);
  for (;;) {
    unsigned first, last;
    int next;
    enum platform_cpu_list_result result = read_cpu(file, &first, &next);
    if (result != PLATFORM_CPU_LIST_OK)
      return result;
    last = first;
    if (next == '-') {
      result = read_cpu(file, &last, &next);
      if (result != PLATFORM_CPU_LIST_OK)
        return result;
    }
    if (last < first)
      return PLATFORM_CPU_LIST_INVALID;
    if (last >= count)
      return PLATFORM_CPU_LIST_TOO_LARGE;
    memset(cpus + first, 1, (size_t)last - first + 1);
    if (next == ',')
      continue;
    if (next == '\n')
      next = fgetc(file);
    return next == EOF && !ferror(file) ? PLATFORM_CPU_LIST_OK
                                        : PLATFORM_CPU_LIST_INVALID;
  }
}

int
platform_pci_numa_node(const char* pci_bus_id)
{
#if defined(__linux__)
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
  return node >= 0 ? node : -1;
#else
  (void)pci_bus_id;
  return -1;
#endif
}

int
platform_numa_cpus(int node, unsigned char* cpus, size_t count)
{
#if defined(__linux__)
  if (node < 0)
    return 1;
  char path[128];
  snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
  FILE* file = fopen(path, "r");
  if (!file) {
    log_debug("NUMA CPU topology unavailable: %s: %s", path, strerror(errno));
    return 1;
  }
  const enum platform_cpu_list_result result =
    platform_parse_cpu_list(file, cpus, count);
  fclose(file);
  if (result == PLATFORM_CPU_LIST_TOO_LARGE)
    log_warn("NUMA placement disabled: %s exceeds the %zu-CPU mask limit",
             path,
             count);
  else if (result != PLATFORM_CPU_LIST_OK)
    log_warn("NUMA placement disabled: invalid CPU list in %s", path);
  return result != PLATFORM_CPU_LIST_OK;
#else
  (void)node;
  (void)cpus;
  (void)count;
  return 1;
#endif
}
