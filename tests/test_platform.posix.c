#include "test_platform.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int
test_tmpdir_create(char* buf, size_t cap)
{
  const char* tmpl = "/tmp/test_XXXXXX";
  size_t len = strlen(tmpl);
  if (len + 1 > cap)
    return -1;
  memcpy(buf, tmpl, len + 1);
  return mkdtemp(buf) ? 0 : -1;
}

int
test_tmpdir_remove(const char* path)
{
  char cmd[4200];
  snprintf(cmd, sizeof(cmd), "rm -rf '%s'", path);
  return system(cmd);
}

int
test_mkdir(const char* path)
{
  if (mkdir(path, 0755) != 0 && errno != EEXIST)
    return -1;
  return 0;
}

int
test_file_exists(const char* path)
{
  struct stat st;
  return stat(path, &st) == 0;
}

struct test_thread
{
  pthread_t handle;
  void (*fn)(void*);
  void* arg;
};

static void*
thread_trampoline(void* p)
{
  struct test_thread* t = (struct test_thread*)p;
  t->fn(t->arg);
  return NULL;
}

int
test_thread_start(struct test_thread** out, void (*fn)(void*), void* arg)
{
  struct test_thread* t = (struct test_thread*)malloc(sizeof(*t));
  if (!t)
    return -1;
  t->fn = fn;
  t->arg = arg;
  if (pthread_create(&t->handle, NULL, thread_trampoline, t) != 0) {
    free(t);
    return -1;
  }
  *out = t;
  return 0;
}

int
test_thread_join(struct test_thread* t)
{
  if (!t)
    return -1;
  int rc = pthread_join(t->handle, NULL);
  free(t);
  return rc == 0 ? 0 : -1;
}
