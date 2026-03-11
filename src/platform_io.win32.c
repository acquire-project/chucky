#include "platform_io.h"

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

  for (size_t i = 1; i < len; ++i) {
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
platform_open_write(const char* path)
{
  return CreateFileA(path,
                     GENERIC_WRITE,
                     0,    // no sharing
                     NULL, // default security
                     CREATE_ALWAYS,
                     FILE_ATTRIBUTE_NORMAL,
                     NULL);
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

void
platform_close(platform_fd fd)
{
  CloseHandle(fd);
}
