#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef HANDLE platform_fd;
#define PLATFORM_FD_INVALID INVALID_HANDLE_VALUE
#else
typedef int platform_fd;
#define PLATFORM_FD_INVALID (-1)
#endif

// Create a single directory. Returns 0 on success or if it already exists.
int platform_mkdir(const char* path);

// Create a directory and all parent directories. Returns 0 on success.
int platform_mkdirp(const char* path);

// Open a file for writing (create/truncate). Returns PLATFORM_FD_INVALID on error.
platform_fd platform_open_write(const char* path);

// Write nbytes at the given byte offset. Returns 0 on success, -1 on error.
int platform_pwrite(platform_fd fd, const void* buf, size_t nbytes, uint64_t offset);

// Sequential write. Returns 0 on success, -1 on error.
int platform_write(platform_fd fd, const void* buf, size_t nbytes);

// Truncate a file to the given size. Returns 0 on success, -1 on error.
int platform_truncate(platform_fd fd, uint64_t size);

// Flags for platform_open_write_ex.
enum { PLATFORM_OPEN_UNBUFFERED = 1 };

// Open a file for writing with flags. Returns PLATFORM_FD_INVALID on error.
platform_fd platform_open_write_ex(const char* path, int flags);

// Close a file descriptor/handle.
void platform_close(platform_fd fd);
