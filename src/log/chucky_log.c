/*
 * Public chucky_log_* API, implemented on top of the rxi logger.
 *
 * The rxi callback signature only carries a single void* udata, so we keep a
 * small table of slots that pair the user's chucky_log_fn with their udata.
 * A single trampoline is registered with rxi per slot; it formats the message
 * into a stack buffer and forwards to the consumer's callback.
 */

#include "chucky_log.h"

#include "log/log.h"

#include <stdarg.h>
#include <stdio.h>

#define CHUCKY_LOG_MAX_CALLBACKS 32
#define CHUCKY_LOG_MSG_BUFFER 2048

struct chucky_log_slot
{
  chucky_log_fn fn;
  void* udata;
  int in_use;
};

static struct chucky_log_slot g_slots[CHUCKY_LOG_MAX_CALLBACKS];

static void
chucky_log_trampoline(log_Event* ev)
{
  struct chucky_log_slot* slot = (struct chucky_log_slot*)ev->udata;
  if (!slot || !slot->fn) {
    return;
  }

  char buf[CHUCKY_LOG_MSG_BUFFER];
  va_list ap;
  va_copy(ap, ev->ap);
  vsnprintf(buf, sizeof(buf), ev->fmt, ap);
  va_end(ap);

  chucky_log_event out = {
    .msg = buf,
    .file = ev->file,
    .line = ev->line,
    .level = (chucky_log_level)ev->level,
  };
  slot->fn(&out, slot->udata);
}

void
chucky_log_set_level(chucky_log_level level)
{
  log_set_level((int)level);
}

void
chucky_log_set_quiet(int quiet)
{
  log_set_quiet(quiet != 0);
}

int
chucky_log_add_callback(chucky_log_fn fn,
                        void* udata,
                        chucky_log_level threshold)
{
  if (!fn) {
    return -1;
  }
  for (int i = 0; i < CHUCKY_LOG_MAX_CALLBACKS; i++) {
    if (!g_slots[i].in_use) {
      g_slots[i].fn = fn;
      g_slots[i].udata = udata;
      g_slots[i].in_use = 1;
      if (log_add_callback(
            chucky_log_trampoline, &g_slots[i], (int)threshold) != 0) {
        g_slots[i].in_use = 0;
        return -1;
      }
      return 0;
    }
  }
  return -1;
}
