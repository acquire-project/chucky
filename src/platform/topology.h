#pragma once

#include <stddef.h>
#include <stdio.h>

enum platform_cpu_list_result
{
  PLATFORM_CPU_LIST_OK,
  PLATFORM_CPU_LIST_INVALID,
  PLATFORM_CPU_LIST_TOO_LARGE,
};

// Parse a sysfs-style CPU list, with an optional final newline. On success,
// cpus[cpu] is 1 for members and 0 otherwise. Contents are unspecified on
// error.
enum platform_cpu_list_result
platform_parse_cpu_list(FILE* file, unsigned char* cpus, size_t count);

// Linux sysfs lookups; unavailable topology returns -1 / nonzero elsewhere.
int
platform_pci_numa_node(const char* pci_bus_id);

int
platform_numa_cpus(int node, unsigned char* cpus, size_t count);
