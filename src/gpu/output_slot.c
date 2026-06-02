#include "gpu/output_slot.h"

#include <string.h>

static int
slot_index_valid(int slot)
{
  return slot == 0 || slot == 1;
}

static int
request_fits_capacity(const struct output_slot_capacity* capacity,
                      const struct output_slot_request* request)
{
  return request->data_bytes <= capacity->data_bytes &&
         request->desc_entries <= capacity->desc_entries &&
         capacity->batch_records > 0;
}

static int
request_fits_slot(const struct output_slot_capacity* capacity,
                  const struct output_slot_entry* slot,
                  const struct output_slot_request* request)
{
  return slot->data_cursor <= capacity->data_bytes &&
         request->data_bytes <= capacity->data_bytes - slot->data_cursor &&
         slot->desc_cursor <= capacity->desc_entries &&
         request->desc_entries <= capacity->desc_entries - slot->desc_cursor &&
         slot->batch_count < capacity->batch_records;
}

static void
fill_append_plan(const struct output_slot_ledger* ledger,
                 const struct output_slot_request* request,
                 int slot,
                 struct output_slot_reservation* out)
{
  const struct output_slot_entry* entry = &ledger->slot[slot];
  out->slot = slot;
  out->data_base = entry->data_cursor;
  out->desc_base = entry->desc_cursor;
  out->batch_index = entry->batch_count;
  out->close_after_append = request->closes_after_append ? 1 : 0;
}

static void
reset_slot(struct output_slot_entry* slot)
{
  memset(slot, 0, sizeof(*slot));
  slot->state = OUTPUT_LEDGER_EMPTY;
}

enum output_ledger_error
output_slot_ledger_init(struct output_slot_ledger* ledger,
                        struct output_slot_capacity capacity)
{
  if (!ledger || capacity.data_bytes == 0 || capacity.desc_entries == 0 ||
      capacity.batch_records == 0)
    return OUTPUT_LEDGER_INVALID;

  memset(ledger, 0, sizeof(*ledger));
  ledger->capacity = capacity;
  ledger->slot[0].state = OUTPUT_LEDGER_EMPTY;
  ledger->slot[1].state = OUTPUT_LEDGER_EMPTY;
  ledger->current = 0;
  ledger->next_close_seq = 1;
  return OUTPUT_LEDGER_OK;
}

enum output_ledger_error
output_slot_ledger_plan_append(const struct output_slot_ledger* ledger,
                               const struct output_slot_request* request,
                               struct output_slot_reservation* out)
{
  if (!ledger || !request || !out || !slot_index_valid(ledger->current))
    return OUTPUT_LEDGER_INVALID;

  memset(out, 0, sizeof(*out));
  out->slot = -1;
  out->close_slot = -1;

  if (!request_fits_capacity(&ledger->capacity, request))
    return OUTPUT_LEDGER_TOO_LARGE;

  const int current = ledger->current;
  const struct output_slot_entry* cur = &ledger->slot[current];
  if (cur->state != OUTPUT_LEDGER_EMPTY && cur->state != OUTPUT_LEDGER_OPEN)
    return OUTPUT_LEDGER_BACKPRESSURE;

  const int current_is_empty = cur->state == OUTPUT_LEDGER_EMPTY;
  const int tail_allows_append = !request->tail_rollforward_blocked ||
                                 current_is_empty || cur->batch_count == 0;
  if (tail_allows_append &&
      request_fits_slot(&ledger->capacity, cur, request)) {
    fill_append_plan(ledger, request, current, out);
    return OUTPUT_LEDGER_OK;
  }

  if (cur->state != OUTPUT_LEDGER_OPEN || cur->batch_count == 0)
    return OUTPUT_LEDGER_BACKPRESSURE;

  const int alternate = current ^ 1;
  const struct output_slot_entry* alt = &ledger->slot[alternate];
  if (alt->state != OUTPUT_LEDGER_EMPTY)
    return OUTPUT_LEDGER_BACKPRESSURE;

  out->close_before_append = 1;
  out->close_slot = current;
  fill_append_plan(ledger, request, alternate, out);
  return OUTPUT_LEDGER_OK;
}

enum output_ledger_error
output_slot_ledger_commit_append(struct output_slot_ledger* ledger,
                                 const struct output_slot_request* request,
                                 const struct output_slot_reservation* plan)
{
  if (!ledger || !request || !plan || !slot_index_valid(plan->slot))
    return OUTPUT_LEDGER_INVALID;

  struct output_slot_entry* slot = &ledger->slot[plan->slot];
  if (slot->state != OUTPUT_LEDGER_EMPTY && slot->state != OUTPUT_LEDGER_OPEN)
    return OUTPUT_LEDGER_INVALID;

  if (plan->data_base != slot->data_cursor ||
      plan->desc_base != slot->desc_cursor ||
      plan->batch_index != slot->batch_count)
    return OUTPUT_LEDGER_INVALID;

  if (!request_fits_slot(&ledger->capacity, slot, request))
    return OUTPUT_LEDGER_TOO_LARGE;

  slot->state = OUTPUT_LEDGER_OPEN;
  slot->data_cursor += request->data_bytes;
  slot->desc_cursor += request->desc_entries;
  slot->batch_count++;
  ledger->current = plan->slot;
  return OUTPUT_LEDGER_OK;
}

enum output_ledger_error
output_slot_ledger_close(struct output_slot_ledger* ledger,
                         int slot,
                         uint64_t* out_seq)
{
  if (!ledger || !slot_index_valid(slot))
    return OUTPUT_LEDGER_INVALID;

  struct output_slot_entry* entry = &ledger->slot[slot];
  if (entry->state != OUTPUT_LEDGER_OPEN || entry->batch_count == 0)
    return OUTPUT_LEDGER_INVALID;

  entry->state = OUTPUT_LEDGER_D2H_IN_FLIGHT;
  entry->close_seq = ledger->next_close_seq++;
  if (slot == ledger->current &&
      ledger->slot[slot ^ 1].state == OUTPUT_LEDGER_EMPTY)
    ledger->current = slot ^ 1;
  if (out_seq)
    *out_seq = entry->close_seq;
  return OUTPUT_LEDGER_OK;
}

enum output_ledger_error
output_slot_ledger_mark_host_ready(struct output_slot_ledger* ledger, int slot)
{
  if (!ledger || !slot_index_valid(slot))
    return OUTPUT_LEDGER_INVALID;

  struct output_slot_entry* entry = &ledger->slot[slot];
  if (entry->state != OUTPUT_LEDGER_D2H_IN_FLIGHT)
    return OUTPUT_LEDGER_INVALID;

  entry->state = OUTPUT_LEDGER_HOST_READY;
  return OUTPUT_LEDGER_OK;
}

enum output_ledger_error
output_slot_ledger_begin_delivery(struct output_slot_ledger* ledger, int slot)
{
  if (!ledger || !slot_index_valid(slot))
    return OUTPUT_LEDGER_INVALID;

  struct output_slot_entry* entry = &ledger->slot[slot];
  if (entry->state != OUTPUT_LEDGER_D2H_IN_FLIGHT &&
      entry->state != OUTPUT_LEDGER_HOST_READY)
    return OUTPUT_LEDGER_INVALID;

  entry->state = OUTPUT_LEDGER_DELIVERING;
  return OUTPUT_LEDGER_OK;
}

enum output_ledger_error
output_slot_ledger_finish_delivery(struct output_slot_ledger* ledger, int slot)
{
  if (!ledger || !slot_index_valid(slot))
    return OUTPUT_LEDGER_INVALID;

  struct output_slot_entry* entry = &ledger->slot[slot];
  if (entry->state != OUTPUT_LEDGER_DELIVERING)
    return OUTPUT_LEDGER_INVALID;

  reset_slot(entry);
  const enum output_ledger_state current_state =
    ledger->slot[ledger->current].state;
  const enum output_ledger_state other_state = ledger->slot[slot ^ 1].state;
  if (other_state == OUTPUT_LEDGER_EMPTY ||
      (current_state != OUTPUT_LEDGER_EMPTY &&
       current_state != OUTPUT_LEDGER_OPEN))
    ledger->current = slot;
  return OUTPUT_LEDGER_OK;
}

enum output_ledger_error
output_slot_ledger_reset_empty(struct output_slot_ledger* ledger, int slot)
{
  if (!ledger || !slot_index_valid(slot))
    return OUTPUT_LEDGER_INVALID;

  reset_slot(&ledger->slot[slot]);
  if (ledger->slot[ledger->current].state != OUTPUT_LEDGER_OPEN)
    ledger->current = slot;
  return OUTPUT_LEDGER_OK;
}

enum output_ledger_error
output_slot_ledger_oldest_closed(const struct output_slot_ledger* ledger,
                                 int* out_slot)
{
  if (!ledger || !out_slot)
    return OUTPUT_LEDGER_INVALID;

  int pick = -1;
  uint64_t pick_seq = UINT64_MAX;
  for (int i = 0; i < 2; ++i) {
    const struct output_slot_entry* entry = &ledger->slot[i];
    if (entry->state != OUTPUT_LEDGER_D2H_IN_FLIGHT &&
        entry->state != OUTPUT_LEDGER_HOST_READY &&
        entry->state != OUTPUT_LEDGER_DELIVERING)
      continue;
    if (entry->close_seq < pick_seq) {
      pick = i;
      pick_seq = entry->close_seq;
    }
  }

  *out_slot = pick;
  return OUTPUT_LEDGER_OK;
}

int
output_slot_ledger_has_work(const struct output_slot_ledger* ledger)
{
  if (!ledger)
    return 0;
  return ledger->slot[0].state != OUTPUT_LEDGER_EMPTY ||
         ledger->slot[1].state != OUTPUT_LEDGER_EMPTY;
}
