#include "chucky_log.h"
#include "util/prelude.h"

#include <string.h>

struct state
{
  int long_msgs_seen;
  int warnings_seen;
};

static void
collect(const chucky_log_event* ev, void* udata)
{
  struct state* st = (struct state*)udata;
  if (strstr(ev->msg, "chucky_log: message truncated"))
    st->warnings_seen++;
  else
    st->long_msgs_seen++;
}

// Emitting a message larger than CHUCKY_LOG_MSG_BUFFER should trigger a WARN
// describing the truncation, but only once even across repeated truncations.
static int
test_truncation_warns_once(void)
{
  log_info("=== test_truncation_warns_once ===");

  chucky_log_set_quiet(1);
  struct state st = { 0 };
  CHECK(Fail, chucky_log_add_callback(collect, &st, CHUCKY_LOG_TRACE) == 0);

  char big[4096];
  memset(big, 'x', sizeof(big) - 1);
  big[sizeof(big) - 1] = '\0';

  log_info("%s", big);
  log_info("%s", big);
  log_info("%s", big);

  chucky_log_set_quiet(0);

  CHECK(Fail, st.long_msgs_seen == 3);
  CHECK(Fail, st.warnings_seen == 1);

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

int
main(void)
{
  return test_truncation_warns_once();
}
