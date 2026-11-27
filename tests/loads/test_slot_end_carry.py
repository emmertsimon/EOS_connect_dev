"""
A running load at the very end of a slot is judged by the slot after it.

27.11., live: the heat pump ran from 23:30, its minimum runtime ended at 01:00, and the
plan in force had it heating without a break until 12:30. The optimizer run that
finished at 01:14:58 had started at 01:14:35, so the slot ending at 01:15 carried only
the 25 s still ahead then - about 11 Wh at 1.6 kW. The gate asked at 01:14:58 and,
never measuring less than a minute, wanted 13 Wh. Stopped, 2 s before a slot the same
plan had at full power. A stopped pump made carrying on a fresh start, so the next plan
put it off to 04:03, where the same happened at 05:44:58 (11 s, 5 Wh). Two starts no
plan asked for.

What these pin: in the last seconds of a slot a *running* load follows the next slot.
It can stop a stop. It must never cause a start, and a real stop stays a stop.
"""

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from src.loads.manager import ManagedLoadManager, ManagedLoadSources

BERLIN = ZoneInfo("Europe/Berlin")
SLOTS = 192
BASE = datetime(2026, 9, 13, 19, 14, 0, tzinfo=BERLIN)
# Slot 76 is 19:00-19:15, slot 77 is 19:15-19:30.
SLOT_76, SLOT_77 = 76, 77
HEAT_PUMP_W = 1600.0


class Clock:
    def __init__(self, now=BASE):
        self.now = now

    def __call__(self):
        return self.now

    def at(self, hour, minute, second=0):
        self.now = BASE.replace(hour=hour, minute=minute, second=second)
        return self.now


class _Journal:
    def __init__(self):
        self.rows = []

    def record_decision(self, load_id, moment, record):
        self.rows.append((load_id, moment, record))
        return True

    def record_sample(self, *_args, **_kwargs):
        return True

    def load_samples(self, *_args, **_kwargs):
        return []

    def load_model_state(self, *_args, **_kwargs):
        return None

    def save_model_state(self, *_args, **_kwargs):
        return True


def make_manager(clock, min_runtime_minutes=0):
    pool = {
        "id": "pool", "type": "pool_heatpump", "enabled": True,
        "temp_sensor": "sensor.pool", "power_sensor": "sensor.pool_power",
        "target_temp": 29.0, "volume_m3": 18.0, "surface_m2": 13.75,
        "rated_power_w": HEAT_PUMP_W, "min_runtime_minutes": min_runtime_minutes,
        "window_start": 3, "window_end": 23, "min_ambient_temp_c": 12.0,
        "max_price_ct_kwh": 25.0, "deadband_k": 0.2,
        "season_start": "04-15", "season_end": "09-30",
    }
    manager = ManagedLoadManager(
        [pool], time_frame_base=900, time_zone=BERLIN, cycle_seconds=300,
        horizon_hours=48,
        sources=ManagedLoadSources(
            read_sensor=lambda name: {
                "sensor.pool": 25.1, "sensor.pool_power": 0.0
            }.get(name),
            price=lambda: [0.00031] * SLOTS,
            feed_in_price=lambda: [0.00008] * SLOTS,
            pv_forecast=lambda: [0.0] * SLOTS,
            base_load=lambda: [400.0] * SLOTS,
            temperature_forecast=lambda: [17.2] * 48,
        ),
        clock=clock,
    )
    manager.external_scheduler = True
    return manager


def schedule(**slots):
    series = [0.0] * SLOTS
    for slot, wh in slots.items():
        series[int(slot[1:])] = wh
    return series


def released(manager):
    return manager.instance("pool").last_release["released"]


def curve_demand():
    """A heat pump on a power curve - the half-of-the-draw threshold of the live case."""
    return SimpleNamespace(
        power_w=[HEAT_PUMP_W] * SLOTS,
        power_at=lambda _index: HEAT_PUMP_W,
        max_power_w=HEAT_PUMP_W,
        detail={"active": True},
        total_wh=20000.0,
        urgent=False,
        reason="",
        store=None,
    )


def gate_at(manager, clock, plan, demand=None):
    """Settle the gate on *plan* at the clock's time, as an adopted schedule would."""
    item = manager.instance("pool")
    ctx = manager._at_now(manager._last_ctx_for["pool"], clock.now)
    return item._gate_on(demand or curve_demand(), ctx, plan)


# --- the live case ---------------------------------------------------------------------

def test_the_live_case_is_below_the_threshold_without_the_carry():
    """The numbers of 01:14:58: 25 s at 1.6 kW against a minute's half-draw."""
    clock = Clock()
    manager = make_manager(clock)
    manager.run_cycle()
    clock.at(19, 14, 58)
    item = manager.instance("pool")
    ctx = manager._at_now(manager._last_ctx_for["pool"], clock.now)
    residue = HEAT_PUMP_W * 25 / 3600.0
    assert residue < item._run_threshold_wh(curve_demand(), ctx)


def test_a_running_heat_pump_is_not_stopped_two_seconds_before_its_next_slot():
    clock = Clock()
    manager = make_manager(clock)
    manager.run_cycle()
    assert gate_at(manager, clock, schedule(s76=400.0, s77=400.0))["released"] is True

    clock.at(19, 14, 58)
    release = gate_at(
        manager, clock, schedule(s76=HEAT_PUMP_W * 25 / 3600.0, s77=400.0)
    )

    assert release["released"] is True
    assert release["reason"] == "planned cheap slot"
    assert release["planned_now"] is True


def test_it_also_carries_across_a_slot_the_plan_left_empty_at_the_very_end():
    """The plan drawn up 11 s before the end - 5 Wh, or nothing at all."""
    clock = Clock()
    manager = make_manager(clock)
    manager.run_cycle()
    gate_at(manager, clock, schedule(s76=400.0, s77=400.0))

    clock.at(19, 14, 59)
    assert gate_at(manager, clock, schedule(s77=400.0))["released"] is True


# --- what it must not do ---------------------------------------------------------------

def test_a_real_stop_at_the_end_of_a_slot_stays_a_stop():
    clock = Clock()
    manager = make_manager(clock)
    manager.run_cycle()
    gate_at(manager, clock, schedule(s76=400.0))

    clock.at(19, 14, 58)
    release = gate_at(manager, clock, schedule(s76=HEAT_PUMP_W * 25 / 3600.0))

    assert release["released"] is False
    assert release["planned_now"] is False


def test_it_never_starts_a_load_early():
    clock = Clock()
    manager = make_manager(clock)
    manager.run_cycle()
    gate_at(manager, clock, schedule())            # stopped

    clock.at(19, 14, 58)
    release = gate_at(manager, clock, schedule(s76=1.5, s77=400.0))

    assert release["released"] is False


def test_further_from_the_end_the_running_slot_decides_as_before():
    """Two minutes before the end is outside the carry: an empty slot stops the load."""
    clock = Clock()
    manager = make_manager(clock)
    manager.run_cycle()
    gate_at(manager, clock, schedule(s76=400.0, s77=400.0))

    clock.at(19, 13, 0)
    assert gate_at(manager, clock, schedule(s77=400.0))["released"] is False


def test_a_next_slot_with_only_residue_does_not_carry():
    clock = Clock()
    manager = make_manager(clock)
    manager.run_cycle()
    gate_at(manager, clock, schedule(s76=400.0, s77=400.0))

    clock.at(19, 14, 58)
    assert gate_at(manager, clock, schedule(s77=0.5))["released"] is False


def test_a_reached_target_still_stops_it():
    clock = Clock()
    manager = make_manager(clock)
    manager.run_cycle()
    gate_at(manager, clock, schedule(s76=400.0, s77=400.0))

    clock.at(19, 14, 58)
    satisfied = curve_demand()
    satisfied.detail = {"active": False}
    release = gate_at(manager, clock, schedule(s77=400.0), demand=satisfied)

    assert release["released"] is False


# --- through the optimizer's path, journal included ------------------------------------

def test_an_adopted_schedule_carries_and_the_journal_says_so():
    clock = Clock()
    manager = make_manager(clock)
    journal = _Journal()
    manager.store = journal
    manager.run_cycle()
    manager.adopt_schedules({"pool": schedule(s76=400.0, s77=400.0)})
    assert released(manager) is True

    clock.at(19, 14, 58)
    manager.adopt_schedules({"pool": schedule(s76=1.2, s77=400.0)})

    assert released(manager) is True
    record = journal.rows[-1][2]
    assert record["released"] is True
    assert record["planned_now"] is True


def test_the_journal_no_longer_calls_a_stopping_slot_planned():
    """27.11. 00:14 UTC read planned_now true on the decision that stopped the pump."""
    clock = Clock()
    manager = make_manager(clock)
    journal = _Journal()
    manager.store = journal
    manager.run_cycle()
    manager.adopt_schedules({"pool": schedule(s76=400.0)})

    clock.at(19, 14, 58)
    manager.adopt_schedules({"pool": schedule(s76=1.2)})

    record = journal.rows[-1][2]
    assert record["released"] is False
    assert record["planned_now"] is False


def test_the_slot_after_the_boundary_is_unchanged():
    """Across the boundary the new running slot decides, with or without the carry."""
    clock = Clock()
    manager = make_manager(clock)
    manager.run_cycle()
    manager.adopt_schedules({"pool": schedule(s76=400.0, s77=400.0)})

    clock.at(19, 15, 30)
    manager.adopt_schedules({"pool": schedule(s77=400.0)})
    assert released(manager) is True

    clock.at(19, 16, 0)
    manager.adopt_schedules({"pool": schedule()})
    assert released(manager) is False
