#include "platform/platform_io.h"

#include <stdio.h>
#include <string.h>

int
platform_mkdir(const char* path)
{
  if (CreateDirectoryA(path, NULL))
    return 0;
  if (GetLastError() == ERROR_ALREADY_EXISTS)
    return 0;
  return -1;
}

int
platform_mkdirp(const char* path)
{
  char tmp[4096];
  size_t len = strlen(path);
  if (len == 0 || len >= sizeof(tmp))
    return -1;
  memcpy(tmp, path, len + 1);

  // Skip drive letter prefix (e.g. "D:\") so we don't try to mkdir "D:"
  size_t start = 1;
  if (len >= 3 && tmp[1] == ':' && (tmp[2] == '\\' || tmp[2] == '/'))
    start = 3;

  for (size_t i = start; i < len; ++i) {
    if (tmp[i] == '/' || tmp[i] == '\\') {
      char saved = tmp[i];
      tmp[i] = '\0';
      if (platform_mkdir(tmp) != 0)
        return -1;
      tmp[i] = saved;
    }
  }
  return platform_mkdir(tmp);
}

platform_fd
platform_open_write(const char* path, int flags)
{
  // PLATFORM_OPEN_UNBUFFERED is a no-op on Windows: NO_BUFFERING + sub-sector
  // EOF corrupts the trailing partial sector on 4K-logical NTFS volumes.
  (void)flags;
  return CreateFileA(
    path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
}

int
platform_pwrite(platform_fd fd, const void* buf, size_t nbytes, uint64_t offset)
{
  const char* p = (const char*)buf;
  size_t remaining = nbytes;
  while (remaining > 0) {
    OVERLAPPED ov = { 0 };
    uint64_t pos = offset + (nbytes - remaining);
    ov.Offset = (DWORD)(pos & 0xFFFFFFFF);
    ov.OffsetHigh = (DWORD)(pos >> 32);

    DWORD to_write = remaining > 0xFFFFFFFF ? 0xFFFFFFFF : (DWORD)remaining;
    DWORD written = 0;
    if (!WriteFile(fd, p, to_write, &written, &ov))
      return -1;
    p += written;
    remaining -= written;
  }
  return 0;
}

int
platform_write(platform_fd fd, const void* buf, size_t nbytes)
{
  const char* p = (const char*)buf;
  size_t remaining = nbytes;
  while (remaining > 0) {
    DWORD to_write = remaining > 0xFFFFFFFF ? 0xFFFFFFFF : (DWORD)remaining;
    DWORD written = 0;
    if (!WriteFile(fd, p, to_write, &written, NULL))
      return -1;
    p += written;
    remaining -= written;
  }
  return 0;
}

int
platform_remove_file(const char* path)
{
  return DeleteFileA(path) ? 0 : -1;
}

// Windows refuses to replace a file another handle has open unless that
// handle allowed delete sharing, which readers generally do not. A reader
// holds it only for one read, so waiting beats failing the write. The bound is
// elapsed time, not attempts: Sleep(1) rounds up to the system timer tick. It
// stays short because an ingest thread is usually waiting on it.
#define RENAME_REPLACE_TIMEOUT_MS 250

int
platform_rename_replace(const char* from, const char* to)
{
  DWORD flags = MOVEFILE_REPLACE_EXISTING;
  ULONGLONG deadline = GetTickCount64() + RENAME_REPLACE_TIMEOUT_MS;
  for (;;) {
    if (MoveFileExA(from, to, flags))
      return 0;
    DWORD err = GetLastError();
    if (err != ERROR_SHARING_VIOLATION && err != ERROR_ACCESS_DENIED)
      return -1;
    if (GetTickCount64() >= deadline)
      return -1;
    Sleep(1);
  }
}

void
platform_close(platform_fd fd)
{
  CloseHandle(fd);
}

int
platform_ftruncate(platform_fd fd, uint64_t logical_size)
{
  FILE_END_OF_FILE_INFO info;
  info.EndOfFile.QuadPart = (LONGLONG)logical_size;
  if (!SetFileInformationByHandle(fd, FileEndOfFileInfo, &info, sizeof(info)))
    return -1;
  return 0;
}

int
platform_path_exists(const char* path)
{
  wchar_t wpath[4096];
  int n = MultiByteToWideChar(
    CP_UTF8, 0, path, -1, wpath, (int)(sizeof(wpath) / sizeof(wpath[0])));
  if (n == 0)
    return -1;

  DWORD attrs = GetFileAttributesW(wpath);
  if (attrs != INVALID_FILE_ATTRIBUTES)
    return 1;

  DWORD err = GetLastError();
  if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND)
    return 0;
  return -1;
}

// Recursive descent: list path's children, delete each (recursing into
// directories), then remove path itself.
static int
remove_tree_recurse(const char* path)
{
  char pattern[4096];
  if ((size_t)snprintf(pattern, sizeof(pattern), "%s\\*", path) >=
      sizeof(pattern))
    return -1;

  WIN32_FIND_DATAA fd;
  HANDLE h = FindFirstFileA(pattern, &fd);
  if (h != INVALID_HANDLE_VALUE) {
    do {
      if (strcmp(fd.cFileName, ".") == 0 || strcmp(fd.cFileName, "..") == 0)
        continue;
      char child[4096];
      if ((size_t)snprintf(
            child, sizeof(child), "%s\\%s", path, fd.cFileName) >=
          sizeof(child)) {
        FindClose(h);
        return -1;
      }
      if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        if (remove_tree_recurse(child) != 0) {
          FindClose(h);
          return -1;
        }
      } else {
        // Clear read-only bit; otherwise DeleteFileA fails with EACCES.
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_READONLY)
          SetFileAttributesA(child, FILE_ATTRIBUTE_NORMAL);
        if (!DeleteFileA(child)) {
          FindClose(h);
          return -1;
        }
      }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
  } else {
    DWORD err = GetLastError();
    if (err != ERROR_FILE_NOT_FOUND && err != ERROR_PATH_NOT_FOUND)
      return -1;
    // Path doesn't exist as a directory; fall through to try as a file.
  }

  if (RemoveDirectoryA(path))
    return 0;
  if (DeleteFileA(path))
    return 0;
  DWORD err = GetLastError();
  if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND)
    return 0;
  return -1;
}

int
platform_remove_tree(const char* path)
{
  if (!path || !path[0])
    return 0;
  return remove_tree_recurse(path);
}
