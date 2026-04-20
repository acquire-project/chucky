// Minimal JSON value validator (no parser; skip-only tokenizer).
#pragma once

#include <stddef.h>

// Returns 1 if s is exactly one well-formed JSON value (with optional
// leading/trailing whitespace). Returns 0 otherwise.
int
json_value_is_valid(const char* s, size_t len);
