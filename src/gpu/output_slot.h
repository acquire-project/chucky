#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum output_ledger_state
  {
    OUTPUT_LEDGER_EMPTY = 0,
    OUTPUT_LEDGER_OPEN = 1,
    OUTPUT_LEDGER_D2H_IN_FLIGHT = 2,
    OUTPUT_LEDGER_HOST_READY = 3,
    OUTPUT_LEDGER_DELIVERING = 4,
  };

  enum output_ledger_error
  {
    OUTPUT_LEDGER_OK = 0,
    OUTPUT_LEDGER_INVALID = 1,
    OUTPUT_LEDGER_TOO_LARGE = 2,
    OUTPUT_LEDGER_BACKPRESSURE = 3,
  };

  struct output_slot_capacity
  {
    size_t data_bytes;
    uint64_t desc_entries;
    uint32_t batch_records;
  };

  struct output_slot_entry
  {
    enum output_ledger_state state;
    size_t data_cursor;
    uint64_t desc_cursor;
    uint32_t batch_count;
    uint64_t close_seq;
  };

  struct output_slot_ledger
  {
    struct output_slot_capacity capacity;
    struct output_slot_entry slot[2];
    int current;
    uint64_t next_close_seq;
  };

  struct output_slot_request
  {
    size_t data_bytes;
    uint64_t desc_entries;
    int closes_after_append;
    int tail_rollforward_blocked;
  };

  struct output_slot_reservation
  {
    int slot;
    size_t data_base;
    uint64_t desc_base;
    uint32_t batch_index;

    int close_before_append;
    int close_slot;
    int close_after_append;
  };

  enum output_ledger_error output_slot_ledger_init(
    struct output_slot_ledger* ledger,
    struct output_slot_capacity capacity);

  enum output_ledger_error output_slot_ledger_plan_append(
    const struct output_slot_ledger* ledger,
    const struct output_slot_request* request,
    struct output_slot_reservation* out);

  enum output_ledger_error output_slot_ledger_commit_append(
    struct output_slot_ledger* ledger,
    const struct output_slot_request* request,
    const struct output_slot_reservation* plan);

  enum output_ledger_error output_slot_ledger_close(
    struct output_slot_ledger* ledger,
    int slot,
    uint64_t* out_seq);

  enum output_ledger_error output_slot_ledger_mark_host_ready(
    struct output_slot_ledger* ledger,
    int slot);

  enum output_ledger_error output_slot_ledger_begin_delivery(
    struct output_slot_ledger* ledger,
    int slot);

  enum output_ledger_error output_slot_ledger_finish_delivery(
    struct output_slot_ledger* ledger,
    int slot);

  enum output_ledger_error output_slot_ledger_reset_empty(
    struct output_slot_ledger* ledger,
    int slot);

  enum output_ledger_error output_slot_ledger_oldest_closed(
    const struct output_slot_ledger* ledger,
    int* out_slot);

  int output_slot_ledger_has_work(const struct output_slot_ledger* ledger);

#ifdef __cplusplus
}
#endif
