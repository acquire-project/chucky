#include "gpu/output_slot.h"
#include "util/prelude.h"

#include <string.h>

static struct output_slot_capacity
capacity(void)
{
  return (struct output_slot_capacity){
    .data_bytes = 100,
    .desc_entries = 10,
    .batch_records = 3,
  };
}

static struct output_slot_request
request(size_t data_bytes, uint64_t desc_entries)
{
  return (struct output_slot_request){
    .data_bytes = data_bytes,
    .desc_entries = desc_entries,
  };
}

static int
check_invariants(const struct output_slot_ledger* ledger)
{
  CHECK(Fail, ledger);
  CHECK(Fail, ledger->current == 0 || ledger->current == 1);
  for (int i = 0; i < 2; ++i) {
    const struct output_slot_entry* slot = &ledger->slot[i];
    CHECK(Fail, slot->data_cursor <= ledger->capacity.data_bytes);
    CHECK(Fail, slot->desc_cursor <= ledger->capacity.desc_entries);
    CHECK(Fail, slot->batch_count <= ledger->capacity.batch_records);
    if (slot->state == OUTPUT_LEDGER_EMPTY) {
      CHECK(Fail, slot->data_cursor == 0);
      CHECK(Fail, slot->desc_cursor == 0);
      CHECK(Fail, slot->batch_count == 0);
      CHECK(Fail, slot->close_seq == 0);
    }
  }
  return 0;
Fail:
  return 1;
}

static int
plan_and_commit(struct output_slot_ledger* ledger,
                struct output_slot_request req,
                struct output_slot_reservation* out)
{
  struct output_slot_reservation plan;
  CHECK(Fail,
        output_slot_ledger_plan_append(ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_commit_append(ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail, check_invariants(ledger) == 0);
  if (out)
    *out = plan;
  return 0;
Fail:
  return 1;
}

static int
test_first_reservation_opens_slot_zero(void)
{
  struct output_slot_ledger ledger;
  struct output_slot_reservation plan;
  struct output_slot_request req = request(20, 2);

  CHECK(Fail, output_slot_ledger_init(&ledger, capacity()) == OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_plan_append(&ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail, plan.slot == 0);
  CHECK(Fail, plan.data_base == 0);
  CHECK(Fail, plan.desc_base == 0);
  CHECK(Fail, plan.batch_index == 0);
  CHECK(Fail, plan.close_before_append == 0);
  CHECK(Fail,
        output_slot_ledger_commit_append(&ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail, ledger.slot[0].state == OUTPUT_LEDGER_OPEN);
  CHECK(Fail, ledger.slot[0].data_cursor == 20);
  CHECK(Fail, ledger.slot[0].desc_cursor == 2);
  CHECK(Fail, ledger.slot[0].batch_count == 1);
  CHECK(Fail, check_invariants(&ledger) == 0);
  return 0;
Fail:
  return 1;
}

static int
test_appends_while_capacity_fits(void)
{
  struct output_slot_ledger ledger;
  struct output_slot_reservation plan;
  CHECK(Fail, output_slot_ledger_init(&ledger, capacity()) == OUTPUT_LEDGER_OK);
  CHECK(Fail, plan_and_commit(&ledger, request(20, 2), NULL) == 0);
  CHECK(Fail, plan_and_commit(&ledger, request(30, 3), &plan) == 0);
  CHECK(Fail, plan.slot == 0);
  CHECK(Fail, plan.data_base == 20);
  CHECK(Fail, plan.desc_base == 2);
  CHECK(Fail, plan.batch_index == 1);
  CHECK(Fail, ledger.slot[0].data_cursor == 50);
  CHECK(Fail, ledger.slot[0].desc_cursor == 5);
  CHECK(Fail, ledger.slot[0].batch_count == 2);
  return 0;
Fail:
  return 1;
}

static int
test_overflow_plans_close_and_alternate(void)
{
  struct output_slot_ledger ledger;
  struct output_slot_reservation plan;
  struct output_slot_request req = request(25, 2);
  CHECK(Fail, output_slot_ledger_init(&ledger, capacity()) == OUTPUT_LEDGER_OK);
  CHECK(Fail, plan_and_commit(&ledger, request(90, 3), NULL) == 0);
  CHECK(Fail,
        output_slot_ledger_plan_append(&ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail, plan.close_before_append == 1);
  CHECK(Fail, plan.close_slot == 0);
  CHECK(Fail, plan.slot == 1);
  CHECK(Fail, plan.data_base == 0);
  CHECK(Fail, plan.desc_base == 0);
  CHECK(Fail, plan.batch_index == 0);
  CHECK(Fail, ledger.slot[0].state == OUTPUT_LEDGER_OPEN);
  CHECK(Fail, check_invariants(&ledger) == 0);
  return 0;
Fail:
  return 1;
}

static int
test_descriptor_overflow_plans_close(void)
{
  struct output_slot_ledger ledger;
  struct output_slot_reservation plan;
  struct output_slot_request req = request(10, 2);
  CHECK(Fail, output_slot_ledger_init(&ledger, capacity()) == OUTPUT_LEDGER_OK);
  CHECK(Fail, plan_and_commit(&ledger, request(10, 9), NULL) == 0);
  CHECK(Fail,
        output_slot_ledger_plan_append(&ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail, plan.close_before_append == 1);
  CHECK(Fail, plan.close_slot == 0);
  CHECK(Fail, plan.slot == 1);
  return 0;
Fail:
  return 1;
}

static int
test_batch_record_overflow_plans_close(void)
{
  struct output_slot_ledger ledger;
  struct output_slot_reservation plan;
  struct output_slot_request req = request(1, 1);
  CHECK(Fail, output_slot_ledger_init(&ledger, capacity()) == OUTPUT_LEDGER_OK);
  CHECK(Fail, plan_and_commit(&ledger, request(1, 1), NULL) == 0);
  CHECK(Fail, plan_and_commit(&ledger, request(1, 1), NULL) == 0);
  CHECK(Fail, plan_and_commit(&ledger, request(1, 1), NULL) == 0);
  CHECK(Fail,
        output_slot_ledger_plan_append(&ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail, plan.close_before_append == 1);
  CHECK(Fail, plan.slot == 1);
  return 0;
Fail:
  return 1;
}

static int
test_tail_rollforward_block_closes_non_empty_current(void)
{
  struct output_slot_ledger ledger;
  struct output_slot_reservation plan;
  struct output_slot_request req = request(10, 1);
  req.tail_rollforward_blocked = 1;
  CHECK(Fail, output_slot_ledger_init(&ledger, capacity()) == OUTPUT_LEDGER_OK);
  CHECK(Fail, plan_and_commit(&ledger, request(10, 1), NULL) == 0);
  CHECK(Fail,
        output_slot_ledger_plan_append(&ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail, plan.close_before_append == 1);
  CHECK(Fail, plan.close_slot == 0);
  CHECK(Fail, plan.slot == 1);
  return 0;
Fail:
  return 1;
}

static int
test_too_large_request_does_not_mutate(void)
{
  struct output_slot_ledger ledger;
  struct output_slot_ledger before;
  struct output_slot_reservation plan;
  struct output_slot_request req = request(101, 1);
  CHECK(Fail, output_slot_ledger_init(&ledger, capacity()) == OUTPUT_LEDGER_OK);
  CHECK(Fail, plan_and_commit(&ledger, request(10, 1), NULL) == 0);
  before = ledger;
  CHECK(Fail,
        output_slot_ledger_plan_append(&ledger, &req, &plan) ==
          OUTPUT_LEDGER_TOO_LARGE);
  CHECK(Fail, memcmp(&before, &ledger, sizeof(ledger)) == 0);
  CHECK(Fail, check_invariants(&ledger) == 0);
  return 0;
Fail:
  return 1;
}

static int
test_backpressure_when_both_slots_occupied(void)
{
  struct output_slot_ledger ledger;
  struct output_slot_reservation plan;
  struct output_slot_request req = request(20, 1);
  struct output_slot_request big_req = request(90, 1);
  CHECK(Fail, output_slot_ledger_init(&ledger, capacity()) == OUTPUT_LEDGER_OK);
  CHECK(Fail, plan_and_commit(&ledger, request(90, 1), NULL) == 0);
  CHECK(Fail,
        output_slot_ledger_plan_append(&ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_close(&ledger, plan.close_slot, NULL) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_commit_append(&ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_plan_append(&ledger, &big_req, &plan) ==
          OUTPUT_LEDGER_BACKPRESSURE);
  CHECK(Fail, check_invariants(&ledger) == 0);
  return 0;
Fail:
  return 1;
}

static int
test_close_sequence_ordering(void)
{
  struct output_slot_ledger ledger;
  struct output_slot_reservation plan;
  struct output_slot_request req = request(20, 1);
  int oldest = -1;
  uint64_t seq0 = 0;
  uint64_t seq1 = 0;
  CHECK(Fail, output_slot_ledger_init(&ledger, capacity()) == OUTPUT_LEDGER_OK);
  CHECK(Fail, plan_and_commit(&ledger, request(90, 1), NULL) == 0);
  CHECK(Fail,
        output_slot_ledger_plan_append(&ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_close(&ledger, plan.close_slot, &seq0) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_commit_append(&ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail, output_slot_ledger_close(&ledger, 1, &seq1) == OUTPUT_LEDGER_OK);
  CHECK(Fail, seq0 < seq1);
  CHECK(Fail,
        output_slot_ledger_oldest_closed(&ledger, &oldest) == OUTPUT_LEDGER_OK);
  CHECK(Fail, oldest == 0);
  CHECK(Fail,
        output_slot_ledger_mark_host_ready(&ledger, 0) == OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_begin_delivery(&ledger, 0) == OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_finish_delivery(&ledger, 0) == OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_oldest_closed(&ledger, &oldest) == OUTPUT_LEDGER_OK);
  CHECK(Fail, oldest == 1);
  return 0;
Fail:
  return 1;
}

static int
test_close_after_append_moves_current_to_empty_alternate(void)
{
  struct output_slot_ledger ledger;
  struct output_slot_reservation plan;
  struct output_slot_request req = request(20, 1);
  struct output_slot_request next_req = request(10, 1);
  req.closes_after_append = 1;
  CHECK(Fail, output_slot_ledger_init(&ledger, capacity()) == OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_plan_append(&ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail, plan.close_after_append == 1);
  CHECK(Fail,
        output_slot_ledger_commit_append(&ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_close(&ledger, plan.slot, NULL) == OUTPUT_LEDGER_OK);
  CHECK(Fail, ledger.current == 1);
  CHECK(Fail,
        output_slot_ledger_plan_append(&ledger, &next_req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail, plan.slot == 1);
  CHECK(Fail, plan.close_before_append == 0);
  return 0;
Fail:
  return 1;
}

static int
test_delivery_completion_reuses_slot(void)
{
  struct output_slot_ledger ledger;
  struct output_slot_reservation plan;
  struct output_slot_request req = request(10, 1);
  CHECK(Fail, output_slot_ledger_init(&ledger, capacity()) == OUTPUT_LEDGER_OK);
  CHECK(Fail, plan_and_commit(&ledger, request(20, 1), NULL) == 0);
  CHECK(Fail, output_slot_ledger_close(&ledger, 0, NULL) == OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_mark_host_ready(&ledger, 0) == OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_begin_delivery(&ledger, 0) == OUTPUT_LEDGER_OK);
  CHECK(Fail,
        output_slot_ledger_finish_delivery(&ledger, 0) == OUTPUT_LEDGER_OK);
  CHECK(Fail, ledger.slot[0].state == OUTPUT_LEDGER_EMPTY);
  CHECK(Fail, output_slot_ledger_has_work(&ledger) == 0);
  CHECK(Fail,
        output_slot_ledger_plan_append(&ledger, &req, &plan) ==
          OUTPUT_LEDGER_OK);
  CHECK(Fail, plan.slot == 0);
  CHECK(Fail, plan.data_base == 0);
  return 0;
Fail:
  return 1;
}

int
main(void)
{
  struct
  {
    const char* name;
    int (*fn)(void);
  } tests[] = {
    { "first_reservation_opens_slot_zero",
      test_first_reservation_opens_slot_zero },
    { "appends_while_capacity_fits", test_appends_while_capacity_fits },
    { "overflow_plans_close_and_alternate",
      test_overflow_plans_close_and_alternate },
    { "descriptor_overflow_plans_close", test_descriptor_overflow_plans_close },
    { "batch_record_overflow_plans_close",
      test_batch_record_overflow_plans_close },
    { "tail_rollforward_block_closes_non_empty_current",
      test_tail_rollforward_block_closes_non_empty_current },
    { "too_large_request_does_not_mutate",
      test_too_large_request_does_not_mutate },
    { "backpressure_when_both_slots_occupied",
      test_backpressure_when_both_slots_occupied },
    { "close_sequence_ordering", test_close_sequence_ordering },
    { "close_after_append_moves_current_to_empty_alternate",
      test_close_after_append_moves_current_to_empty_alternate },
    { "delivery_completion_reuses_slot", test_delivery_completion_reuses_slot },
  };

  int rc = 0;
  for (size_t i = 0; i < countof(tests); ++i) {
    int r = tests[i].fn();
    if (r) {
      log_error("FAIL: %s", tests[i].name);
      rc = 1;
    } else {
      log_info("PASS: %s", tests[i].name);
    }
  }
  return rc;
}
