#include "platform_cmd.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>

int
platform_cmd_run(const char* cmd)
{
  int rc = system(cmd);
  if (rc == -1)
    return -1;
  if (!WIFEXITED(rc))
    return -1;
  return WEXITSTATUS(rc);
}

int
platform_cmd_capture(const char* cmd, uint8_t** out, size_t* out_len)
{
  FILE* f = popen(cmd, "r");
  if (!f)
    return -1;

  size_t cap = 4096;
  size_t len = 0;
  uint8_t* buf = (uint8_t*)malloc(cap);
  if (!buf) {
    pclose(f);
    return -1;
  }

  for (;;) {
    size_t n = fread(buf + len, 1, cap - len, f);
    len += n;
    if (n == 0)
      break;
    if (len == cap) {
      cap *= 2;
      uint8_t* tmp = (uint8_t*)realloc(buf, cap);
      if (!tmp) {
        free(buf);
        pclose(f);
        return -1;
      }
      buf = tmp;
    }
  }

  int rc = pclose(f);
  if (!WIFEXITED(rc) || WEXITSTATUS(rc) != 0) {
    free(buf);
    return -1;
  }

  *out = buf;
  *out_len = len;
  return 0;
}
