#include "platform_io.h"

#include "log/log.h"
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int
platform_mkdir(const char* path)
{
  if (mkdir(path, 0755) != 0 && errno != EEXIST)
    return -1;
  return 0;
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
    if (tmp[i] == '/') {
      tmp[i] = '\0';
      if (platform_mkdir(tmp) != 0)
        return -1;
      tmp[i] = '/';
    }
  }
  return platform_mkdir(tmp);
}

platform_fd
platform_open_write(const char* path)
{
  return platform_open_write_ex(path, 0);
}

platform_fd
platform_open_write_ex(const char* path, int flags)
{
  if (flags & PLATFORM_OPEN_UNBUFFERED) {
    log_error("Unbuffered IO (O_DIRECT) is not yet implemented on POSIX");
    return -1;
  }
  return open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
}

int
platform_pwrite(platform_fd fd, const void* buf, size_t nbytes, uint64_t offset)
{
  const char* p = (const char*)buf;
  size_t remaining = nbytes;
  while (remaining > 0) {
    ssize_t n = pwrite(fd, p, remaining, (off_t)(offset + (nbytes - remaining)));
    if (n < 0)
      return -1;
    p += n;
    remaining -= (size_t)n;
  }
  return 0;
}

int
platform_write(platform_fd fd, const void* buf, size_t nbytes)
{
  const char* p = (const char*)buf;
  size_t remaining = nbytes;
  while (remaining > 0) {
    ssize_t n = write(fd, p, remaining);
    if (n < 0)
      return -1;
    p += n;
    remaining -= (size_t)n;
  }
  return 0;
}

int
platform_truncate(platform_fd fd, uint64_t size)
{
  return ftruncate(fd, (off_t)size);
}

void
platform_close(platform_fd fd)
{
  close(fd);
}
