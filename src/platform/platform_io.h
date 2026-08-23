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
int
platform_mkdir(const char* path);

// Create a directory and all parent directories. Returns 0 on success.
int
platform_mkdirp(const char* path);

// Flags for platform_open_write.
enum
{
  PLATFORM_OPEN_UNBUFFERED = 1
};

// Open a file for writing (create/truncate). Returns PLATFORM_FD_INVALID on
// error.
platform_fd
platform_open_write(const char* path, int flags);

// Write nbytes at the given byte offset. Returns 0 on success, -1 on error.
int
platform_pwrite(platform_fd fd,
                const void* buf,
                size_t nbytes,
                uint64_t offset);

// Sequential write. Returns 0 on success, -1 on error.
int
platform_write(platform_fd fd, const void* buf, size_t nbytes);

// Delete a single file. Returns 0 on success, -1 on error.
int
platform_remove_file(const char* path);

// Replaces to if it exists. Atomic for readers of to, as long as both paths
// are on the same filesystem. Returns 0 on success, -1 on error.
int
platform_rename_replace(const char* from, const char* to);

// Close a file descriptor/handle.
void
platform_close(platform_fd fd);

// Truncate a file to logical_size bytes. Returns 0 on success, -1 on error.
int
platform_ftruncate(platform_fd fd, uint64_t logical_size);

// Non-zero when setting a file's size up front stops later writes from
// extending it. Not so on NTFS, where only writing moves a file's valid data
// length.
int
platform_presize_helps(void);

// Returns 1 if path exists, 0 if it does not (ENOENT / path-component-is-file
// on Windows / not-a-dir), -1 on unexpected IO error.
int
platform_path_exists(const char* path);

// Recursively remove the file or directory at path, including all contents.
// A missing path is treated as success. Returns 0 on success, -1 if any entry
// could not be removed.
int
platform_remove_tree(const char* path);
