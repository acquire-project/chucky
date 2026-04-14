#pragma once

#include <stddef.h>

#ifdef _WIN32
#define NULL_DEV "NUL"
#else
#define NULL_DEV "/dev/null"
#endif

// Create a unique temporary directory. Writes the path into buf.
// Returns 0 on success, -1 on error.
int
test_tmpdir_create(char* buf, size_t cap);

// Recursively remove a directory tree (rm -rf equivalent).
int
test_tmpdir_remove(const char* path);

// Create a single directory (no mode argument — ignored on Windows).
int
test_mkdir(const char* path);

// Check whether a file exists. Returns 1 if it exists, 0 otherwise.
int
test_file_exists(const char* path);
