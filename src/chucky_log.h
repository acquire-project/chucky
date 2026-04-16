#pragma once

/*
 * Public logging control for chucky consumers.
 *
 * Use these to forward a caller-supplied log level into chucky, intercept
 * messages with a callback for routing into a host logging framework, or
 * silence the default stderr sink.
 *
 * Callbacks fire on whichever thread produced the log line. Logger state is
 * process-global; there is no per-stream scoping.
 */

#include <time.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef enum chucky_log_level
  {
    CHUCKY_LOG_TRACE = 0,
    CHUCKY_LOG_DEBUG,
    CHUCKY_LOG_INFO,
    CHUCKY_LOG_WARN,
    CHUCKY_LOG_ERROR,
    CHUCKY_LOG_FATAL,
  } chucky_log_level;

  // Pointers are valid only for the duration of the callback; copy anything
  // that needs to outlive it.
  typedef struct chucky_log_event
  {
    const char* msg;
    const char* file;
    int line;
    chucky_log_level level;
    time_t time;
  } chucky_log_event;

  typedef void (*chucky_log_fn)(const chucky_log_event* ev, void* udata);

  // Drop events below level before they reach any sink.
  void chucky_log_set_level(chucky_log_level level);

  // Suppress the default stderr sink. Registered callbacks keep firing.
  void chucky_log_set_quiet(int quiet);

  // Returns 0 on success, non-zero if the callback table is full.
  int chucky_log_add_callback(chucky_log_fn fn,
                              void* udata,
                              chucky_log_level threshold);

#ifdef __cplusplus
}
#endif
