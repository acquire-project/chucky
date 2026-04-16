#pragma once

/*
 * Public logging control for chucky consumers.
 *
 * Chucky emits log messages internally through the rxi logger (src/log/log.h,
 * which remains a private header). Consumers that need to control the log
 * threshold or route messages somewhere other than stderr use the small API
 * declared here.
 *
 * Typical uses:
 *   - Forward a caller-supplied log level into chucky.
 *   - Intercept messages with a callback so they can be routed to the host
 *     application's logging framework (e.g. Python's ``logging`` module).
 *   - Silence the default stderr sink when a callback is installed.
 *
 * Thread-safety: callbacks fire on whichever thread produced the log line.
 * Consumers are responsible for any synchronization they need (e.g. a Python
 * sink should acquire the GIL on entry).
 *
 * Scope: chucky's logger state is process-global. Per-stream or per-sink
 * scoping is not supported.
 */

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

  /*
   * A log record passed to a consumer-supplied callback. The ``msg`` field is
   * already formatted by chucky — callbacks do not need to handle va_list.
   * The pointers are only valid for the duration of the callback call; copy
   * anything that needs to outlive the call.
   */
  typedef struct chucky_log_event
  {
    const char* msg;
    const char* file;
    int line;
    chucky_log_level level;
  } chucky_log_event;

  typedef void (*chucky_log_fn)(const chucky_log_event* ev, void* udata);

  /*
   * Set the minimum level that will be emitted. Messages below ``level`` are
   * dropped before they reach the default stderr sink or any registered
   * callback.
   */
  void chucky_log_set_level(chucky_log_level level);

  /*
   * Suppress the default stderr sink. Registered callbacks continue to fire.
   * Pass a non-zero value to enable quiet mode, zero to re-enable stderr.
   */
  void chucky_log_set_quiet(int quiet);

  /*
   * Register ``fn`` to receive every log event at or above ``threshold``.
   * ``udata`` is passed through unchanged on every call. Returns 0 on
   * success, non-zero if the callback table is full.
   */
  int chucky_log_add_callback(chucky_log_fn fn,
                              void* udata,
                              chucky_log_level threshold);

#ifdef __cplusplus
}
#endif
