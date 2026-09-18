#pragma once

struct platform_placement;

struct platform_placement_scope
{
  unsigned long saved_affinity[128 / sizeof(unsigned long)];
  int restore_affinity;
  int restore_memory_policy;
};

int
platform_pci_numa_node(const char* pci_bus_id);

struct platform_placement*
platform_placement_create(int node);

void
platform_placement_destroy(struct platform_placement* placement);

void
platform_placement_enter(const struct platform_placement* placement,
                         int place_memory,
                         struct platform_placement_scope* scope);

int
platform_placement_leave(struct platform_placement_scope* scope);
