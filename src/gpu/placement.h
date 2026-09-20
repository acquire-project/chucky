#pragma once

#include "platform/placement.h"

struct platform_placement*
gpu_placement_create(void);

int
gpu_placement_leave(struct platform_placement_scope* scope);
