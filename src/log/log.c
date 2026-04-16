#include "log/log.h"
#include "chucky_log.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define CHUCKY_LOG_MAX_CALLBACKS 32
#define CHUCKY_LOG_MSG_BUFFER 2048

struct callback
{
  chucky_log_fn fn;
  void* udata;
  chucky_log_level threshold;
  int in_use;
};

static struct
{
  chucky_log_level level;
  int quiet;
  int truncation_warned;
  struct callback callbacks[CHUCKY_LOG_MAX_CALLBACKS];
} L;

static const char* level_strings[] = {
  "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL",
};

static void
stderr_sink(const chucky_log_event* ev, void* udata)
{
  (void)udata;
  char timebuf[16];
  struct tm* tm = localtime(&ev->time);
  timebuf[strftime(timebuf, sizeof(timebuf), "%H:%M:%S", tm)] = '\0';
  fprintf(stderr,
          "%s %-5s %s:%d: %s\n",
          timebuf,
          level_strings[ev->level],
          ev->file,
          ev->line,
          ev->msg);
  fflush(stderr);
}

void
chucky_log_set_level(chucky_log_level level)
{
  L.level = level;
}

void
chucky_log_set_quiet(int quiet)
{
  L.quiet = quiet != 0;
}

int
chucky_log_add_callback(chucky_log_fn fn,
                        void* udata,
                        chucky_log_level threshold)
{
  if (!fn)
    return -1;
  for (int i = 0; i < CHUCKY_LOG_MAX_CALLBACKS; i++) {
    if (!L.callbacks[i].in_use) {
      L.callbacks[i].fn = fn;
      L.callbacks[i].udata = udata;
      L.callbacks[i].threshold = threshold;
      L.callbacks[i].in_use = 1;
      return 0;
    }
  }
  return -1;
}

void
log_log(int level, const char* file, int line, const char* fmt, ...)
{
  if (level < (int)L.level)
    return;

  char buf[CHUCKY_LOG_MSG_BUFFER];
  va_list ap;
  va_start(ap, fmt);
  int needed = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);

  chucky_log_event ev = {
    .msg = buf,
    .file = file,
    .line = line,
    .level = (chucky_log_level)level,
    .time = time(NULL),
  };

  if (!L.quiet)
    stderr_sink(&ev, NULL);

  for (int i = 0; i < CHUCKY_LOG_MAX_CALLBACKS; i++) {
    const struct callback* cb = &L.callbacks[i];
    if (cb->in_use && ev.level >= cb->threshold)
      cb->fn(&ev, cb->udata);
  }

  // First-time truncation notice. Recursion is bounded: the warning message
  // is short, the flag is set before recursing, and L.truncation_warned gates
  // any subsequent entry.
  if (needed >= (int)sizeof(buf) && !L.truncation_warned) {
    L.truncation_warned = 1;
    log_warn("chucky_log: message truncated at %d bytes "
             "(increase CHUCKY_LOG_MSG_BUFFER)",
             (int)sizeof(buf));
  }
}
