#pragma once

#include <stddef.h>
#include <stdint.h>

// Run a shell command. Returns 0 if the command exits with status 0.
int
platform_cmd_run(const char* cmd);

// Run a shell command and capture its stdout.
// On success, sets *out to a malloc'd buffer and *out_len to its length.
// Returns 0 if the command exits with status 0. Caller must free(*out).
int
platform_cmd_capture(const char* cmd, uint8_t** out, size_t* out_len);
