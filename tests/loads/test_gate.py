"""Release gating: minimum runtime, overrides and the reasons reported."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.loads.gate import (
    OVERRIDE_BLOCK,
    OVERRIDE_RELEASE,
    REASON_MIN_RUNTIME,
    REASON_NO_DEMAND,
    REASON_NOT_PLANNED,
    REASON_OVERRIDE,
    REASON_PLANNED,
    REASON_URGENT,
    ReleaseGate,
)

BERLIN = ZoneInfo("Europe/Berlin")
T0 = datetime(2026, 6, 1, 12, 0, tzinfo=BERLIN)


def _at(minutes):
    return T0 + timedelta(minutes=minutes)


def test_a_planned_slot_releases():
    gate = ReleaseGate("pool")
    result = gate.evaluate(planned=True, urgent=False, now=T0)
    assert result["released"] is True
    assert result["reason"] == REASON_PLANNED


def test_an_unplanned_slot_blocks():
    gate = ReleaseGate("pool")
    result = gate.evaluate(planned=False, urgent=False, now=T0)
    assert result["released"] is False
    assert result["reason"] == REASON_NOT_PLANNED


def test_minimum_runtime_holds_a_release_across_a_replan():
    """
    The plan is recomputed every cycle; without the hold a 15-minute reshuffle would
    cycle the compressor.
    """
    gate = ReleaseGate("pool", min_runtime_minutes=30)
    gate.evaluate(planned=True, urgent=False, now=T0)

    held = gate.evaluate(planned=False, urgent=False, now=_at(10))
    assert held["released"] is True
    assert held["reason"] == REASON_MIN_RUNTIME

    expired = gate.evaluate(planned=False, urgent=False, now=_at(31))
    assert expired["released"] is False


def test_minimum_runtime_does_not_survive_the_target_being_reached():
    """Holding a pump on against a satisfied setpoint overshoots the target."""
    gate = ReleaseGate("pool", min_runtime_minutes=30)
    gate.evaluate(planned=True, urgent=False, now=T0)
    result = gate.evaluate(planned=False, urgent=False, now=_at(5), has_demand=False)
    assert result["released"] is False
    assert result["reason"] == REASON_NO_DEMAND


def test_no_demand_blocks_even_in_a_planned_slot():
    gate = ReleaseGate("pool")
    result = gate.evaluate(planned=True, urgent=False, now=T0, has_demand=False)
    assert result["released"] is False
    assert result["reason"] == REASON_NO_DEMAND


def test_urgent_releases_regardless_of_the_plan():
    gate = ReleaseGate("pool")
    result = gate.evaluate(planned=False, urgent=True, now=T0)
    assert result["released"] is True
    assert result["reason"] == REASON_URGENT


def test_override_release_beats_an_unplanned_slot():
    gate = ReleaseGate("pool")
    gate.set_override(OVERRIDE_RELEASE, 60, T0)
    result = gate.evaluate(planned=False, urgent=False, now=_at(5))
    assert result["released"] is True
    assert result["reason"] == REASON_OVERRIDE


def test_override_block_beats_the_minimum_runtime_hold():
    """An override the model can veto is not an override."""
    gate = ReleaseGate("pool", min_runtime_minutes=60)
    gate.evaluate(planned=True, urgent=False, now=T0)
    gate.set_override(OVERRIDE_BLOCK, 30, _at(5))
    result = gate.evaluate(planned=True, urgent=True, now=_at(6))
    assert result["released"] is False
    assert result["reason"] == REASON_OVERRIDE


def test_override_expires_and_control_returns_to_the_plan():
    gate = ReleaseGate("pool")
    gate.set_override(OVERRIDE_BLOCK, 10, T0)
    assert gate.evaluate(planned=True, urgent=False, now=_at(5))["released"] is False
    assert gate.evaluate(planned=True, urgent=False, now=_at(11))["released"] is True


def test_override_can_be_cleared_explicitly():
    gate = ReleaseGate("pool")
    gate.set_override(OVERRIDE_BLOCK, 60, T0)
    gate.set_override(None, 0, T0)
    assert gate.evaluate(planned=True, urgent=False, now=_at(1))["released"] is True


def test_an_unknown_override_mode_is_rejected():
    with pytest.raises(ValueError):
        ReleaseGate("pool").set_override("maybe", 10, T0)


def test_released_since_is_set_on_start_and_cleared_on_stop():
    gate = ReleaseGate("pool")
    started = gate.evaluate(planned=True, urgent=False, now=T0)
    assert started["released_since"] == T0.isoformat()
    stopped = gate.evaluate(planned=False, urgent=False, now=_at(20))
    assert stopped["released_since"] is None


def test_released_since_does_not_creep_forward_while_the_run_continues():
    """
    The property the whole minimum-runtime commitment rests on, and it is load-bearing
    in a way that is easy to break by accident.

    `commitment` pins the head of the horizon for whatever is left of the minimum,
    measured from `released_since`. If that timestamp were refreshed on every
    evaluation the remainder would never reach zero, the pin would never lift, and a
    running load could never be stopped - it would keep itself released, and the next
    plan would pin it again. That latch ran a live pool through an evening peak into a
    34.65 ct slot under a 30 ct limit before it was caught.

    Nothing here would fail loudly if it regressed: the load would simply never stop.
    """
    gate = ReleaseGate("pool", min_runtime_minutes=15)
    first = gate.evaluate(planned=True, urgent=False, now=T0)

    for minute in (1, 5, 20, 90, 600):
        again = gate.evaluate(planned=True, urgent=False, now=_at(minute))
        assert again["released"] is True
        assert again["released_since"] == first["released_since"], (
            f"released_since moved at +{minute} min - the minimum would never expire"
        )


def test_a_fresh_release_after_a_stop_does_restart_the_clock():
    """The other half: a genuinely new run gets its own minimum."""
    gate = ReleaseGate("pool", min_runtime_minutes=15)
    gate.evaluate(planned=True, urgent=False, now=T0)
    gate.evaluate(planned=False, urgent=False, now=_at(20))
    restarted = gate.evaluate(planned=True, urgent=False, now=_at(40))
    assert restarted["released_since"] == _at(40).isoformat()


def test_status_reports_without_changing_the_decision():
    gate = ReleaseGate("pool", min_runtime_minutes=15)
    gate.evaluate(planned=True, urgent=False, now=T0)
    status = gate.status()
    assert status["released"] is True
    assert status["min_runtime_minutes"] == 15
    assert gate.released is True


# --- after a restart ---------------------------------------------------------------------

def _journal(*entries):
    """(minutes before T0, released) pairs, returned newest first like the store."""
    rows = [{"timestamp": _at(-minutes).isoformat(), "released": released}
            for minutes, released in entries]
    return sorted(rows, key=lambda row: row["timestamp"], reverse=True)


def _every_three(newest, oldest, released):
    """A stretch of the journal as it is written: one decision every three minutes."""
    return [(minutes, released) for minutes in range(newest, oldest + 1, 3)]


def test_a_run_in_progress_is_restored_with_its_start():
    gate = ReleaseGate("house", min_runtime_minutes=90)
    journal = _journal(*_every_three(1, 40, True), *_every_three(43, 61, False))
    assert gate.restore(journal, T0, max_age_s=1200) is True
    assert gate.released is True
    assert gate.released_since == _at(-40)
    # 40 of 90 minutes served: the hold carries the run over a plan that drops it.
    result = gate.evaluate(planned=False, urgent=False, now=_at(10))
    assert result["released"] is True
    assert result["reason"] == REASON_MIN_RUNTIME


def test_a_restored_run_ends_when_its_minimum_is_served():
    gate = ReleaseGate("house", min_runtime_minutes=90)
    gate.restore(_journal(*_every_three(1, 40, True)), T0, max_age_s=1200)
    assert gate.evaluate(planned=False, urgent=False, now=_at(51))["released"] is False


def test_a_rest_is_restored_with_its_start():
    gate = ReleaseGate("house", min_runtime_minutes=90)
    gate.restore(_journal(*_every_three(3, 30, False), (33, True)), T0, max_age_s=1200)
    assert gate.released is False
    assert gate.blocked_since == _at(-30)
    assert gate.released_since is None


def test_a_stale_journal_restores_nothing():
    gate = ReleaseGate("house", min_runtime_minutes=90)
    assert gate.restore(_journal((25, True), (40, True)), T0, max_age_s=1200) is False
    assert gate.released is False


def test_a_hole_in_the_journal_ends_the_run_it_describes():
    """EOS was down for an hour before: the run behind the gap is not this one."""
    gate = ReleaseGate("house", min_runtime_minutes=90)
    gate.restore(_journal((1, True), (5, True), (70, True), (75, True)), T0,
                 max_age_s=1200)
    assert gate.released_since == _at(-5)


def test_an_empty_or_unreadable_journal_restores_nothing():
    gate = ReleaseGate("house")
    assert gate.restore([], T0, max_age_s=1200) is False
    assert gate.restore([{"timestamp": "garbage", "released": True}], T0,
                        max_age_s=1200) is False


def test_an_appliance_drawing_power_is_taken_as_running():
    gate = ReleaseGate("house", min_runtime_minutes=90)
    assert gate.adopt_running(2100.0, 50.0) is True
    assert gate.released is True
    # No start time is invented: no hold, the plan decides at once.
    assert gate.evaluate(planned=False, urgent=False, now=T0)["released"] is False


def test_an_idle_appliance_is_not_taken_as_running():
    gate = ReleaseGate("house")
    assert gate.adopt_running(20.0, 50.0) is False
    assert gate.adopt_running(None, 50.0) is False
    assert gate.released is False
