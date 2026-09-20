#pragma once

struct platform_placement;

struct platform_placement_scope
{
  unsigned long saved_affinity[128 / sizeof(unsigned long)];
  int restore_affinity;
  int restore_memory_policy;
};

// A placement is immutable and may be shared by threads. Missing topology
// returns NULL, which enter treats as a no-op.
struct platform_placement*
platform_placement_create(int node);

void
platform_placement_destroy(struct platform_placement* placement);

// Restrict this thread to allowed local CPUs. With place_memory, prefer the
// node only when the caller has the default memory policy. Best effort: an
// unavailable operation preserves that part of the caller's placement.
// Each enter needs its own scope; leave nested scopes in reverse order on
// the same thread. The placement may be destroyed before leaving the scope.
void
platform_placement_enter(const struct platform_placement* placement,
                         int place_memory,
                         struct platform_placement_scope* scope);

// Restore changes and consume the scope, even on failure. Returns nonzero if
// restoration failed. A zeroed or already-left scope is a no-op.
int
platform_placement_leave(struct platform_placement_scope* scope);
