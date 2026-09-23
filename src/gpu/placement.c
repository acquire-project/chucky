#include "gpu/placement.h"

#include "log/log.h"
#include "platform/topology.h"

#include <cuda.h>
#include <stddef.h>

struct platform_placement*
gpu_placement_create(void)
{
  CUdevice device;
  char pci_bus_id[32];
  if (cuCtxGetDevice(&device) != CUDA_SUCCESS ||
      cuDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device) !=
        CUDA_SUCCESS)
    return NULL;
  return platform_placement_create(platform_pci_numa_node(pci_bus_id));
}

int
gpu_placement_leave(struct platform_placement_scope* scope)
{
  const int failed = platform_placement_leave(scope);
  if (failed)
    log_error("could not restore the caller's CPU affinity or memory policy");
  return failed;
}
