"""
Two midnights for a few minutes after the day turns.

Every per-slot array in EOS counts from a local midnight. The managed-load module takes
its midnight from its own five-minute cycle, the optimizer from the moment its run
starts. Between 00:00 and whichever of the two moves second, they count from different
days. A schedule from a 00:03 run then arrives on today's grid while the module's
context still counts from yesterday: the running slot reads as slot 96, nothing is
planned there, and the load's release drops until the next cycle. The request has the
same fault in the other direction: a record built at 23:59 rotated by a 00:03 clock
hands the solver yesterday's air temperatures and COPs.

What these pin: whichever side of midnight a plan or a record was made, it is read on
the grid of whoever reads it.
"""

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from src.loads.manager import ManagedLoadManager, ManagedLoadSources
from src.timeseries_grid import reanchor, slots_between

BERLIN = ZoneInfo("Europe/Berlin")
SLOTS = 192
YESTERDAY = datetime(2026, 11, 19, tzinfo=BERLIN)
TODAY = datetime(2026, 11, 20, tzinfo=BERLIN)


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def make_manager(clock, min_runtime_minutes=0):
    """A load with no window or season, so midnight is an ordinary slot for it."""
    pool = {
        "id": "pool", "type": "pool_heatpump", "enabled": True,
        "temp_sensor": "sensor.pool", "power_sensor": "sensor.pool_power",
        "target_temp": 29.0, "volume_m3": 18.0, "surface_m2": 13.75,
        "rated_power_w": 1600.0, "min_runtime_minutes": min_runtime_minutes,
        "window_start": None, "window_end": None, "min_ambient_temp_c": None,
        "season_start": None, "season_end": None, "deadband_k": 0.2,
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


def schedule_for(*slots):
    series = [0.0] * SLOTS
    for slot in slots:
        series[slot] = 400.0
    return series


def release(manager):
    return manager.instance("pool").last_release


# --- the grid helpers ------------------------------------------------------------------

def test_an_ordinary_night_is_96_quarter_hours_and_a_dst_night_is_not():
    assert slots_between(YESTERDAY, TODAY, 900) == 96
    # The clocks go back on 25 October: that day is 25 hours long, the spring one 23.
    assert slots_between(
        datetime(2026, 10, 25, tzinfo=BERLIN), datetime(2026, 10, 26, tzinfo=BERLIN), 900
    ) == 100
    assert slots_between(
        datetime(2026, 3, 29, tzinfo=BERLIN), datetime(2026, 3, 30, tzinfo=BERLIN), 900
    ) == 92
    # An anchor read back from ISO text carries a fixed offset and still agrees.
    assert slots_between(
        datetime.fromisoformat(YESTERDAY.isoformat()), TODAY, 900
    ) == 96
    assert slots_between(None, TODAY, 900) == 0


def test_a_naive_and_an_aware_anchor_are_compared_as_wall_time():
    """A part of the program without a time zone counts naive local midnights."""
    naive_today = TODAY.replace(tzinfo=None)
    assert slots_between(YESTERDAY, naive_today, 900) == 96
    assert slots_between(naive_today, YESTERDAY, 900) == -96
    assert slots_between(naive_today, TODAY, 900) == 0


def test_reanchoring_keeps_the_length_and_moves_the_hours():
    series = list(range(8))
    assert reanchor(series, 3, 0) == [3, 4, 5, 6, 7, 0, 0, 0]
    assert reanchor(series, -2, 0) == [0, 0, 0, 1, 2, 3, 4, 5]
    assert reanchor(series, 0, 0) == series
    assert reanchor(series, 2, 9, length=4) == [2, 3, 4, 5]


# --- the schedule coming back ----------------------------------------------------------

def test_a_schedule_on_todays_grid_releases_a_context_from_yesterday():
    """Context from 23:59, schedule from the 00:03 run."""
    clock = Clock(YESTERDAY.replace(hour=23, minute=59))
    manager = make_manager(clock)
    manager.run_cycle()

    clock.now = TODAY.replace(minute=3)
    manager.adopt_schedules(
        {"pool": schedule_for(0, 1)}, anchor=TODAY.isoformat()
    )

    item = manager.instance("pool")
    assert release(manager)["released"] is True
    assert release(manager)["reason"] == "planned cheap slot"
    # Kept on the context's grid: 00:00 today is slot 96 of yesterday's.
    assert item.last_plan[96] == pytest.approx(400.0)
    assert item.last_plan[0] == 0.0


def test_without_the_anchor_the_same_schedule_misses_the_running_slot():
    """What the house saw before - kept so the test above cannot pass by accident."""
    clock = Clock(YESTERDAY.replace(hour=23, minute=59))
    manager = make_manager(clock)
    manager.run_cycle()

    clock.now = TODAY.replace(minute=3)
    manager.adopt_schedules({"pool": schedule_for(0, 1)})

    assert release(manager)["released"] is False
    assert manager.instance("pool").last_plan[96] == 0.0   # the running slot


def test_a_schedule_from_a_run_that_started_before_midnight_still_lands():
    """The other order: the 23:59:58 run answers after the context has moved on."""
    clock = Clock(TODAY.replace(minute=0, second=30))
    manager = make_manager(clock)
    manager.run_cycle()                         # context already on today's grid

    manager.adopt_schedules(
        {"pool": schedule_for(96)}, anchor=YESTERDAY.isoformat()
    )

    item = manager.instance("pool")
    assert release(manager)["released"] is True
    assert item.last_plan[0] == pytest.approx(400.0)


# --- the plan the module keeps ---------------------------------------------------------

def test_a_kept_schedule_moves_with_the_context_at_midnight():
    """
    The optimizer's plan lives on across cycles. The first cycle after midnight moves
    the context to today's grid, and the plan has to come along.
    """
    clock = Clock(YESTERDAY.replace(hour=23, minute=59))
    manager = make_manager(clock)
    manager.run_cycle()
    manager.adopt_schedules(
        {"pool": schedule_for(96, 97)}, anchor=YESTERDAY.isoformat()
    )
    item = manager.instance("pool")
    assert item.last_plan[96] == pytest.approx(400.0)

    clock.now = TODAY.replace(minute=1)
    manager.run_cycle()

    assert item.last_plan[0] == pytest.approx(400.0)
    assert item.last_plan[1] == pytest.approx(400.0)
    assert len(item.last_plan) == SLOTS
    # The slot now running, on the context the next schedule is read against.
    ctx = manager._last_ctx_for["pool"]  # pylint: disable=protected-access
    assert ctx.current_slot == 0
    assert item.last_plan[ctx.current_slot] == pytest.approx(400.0)


def test_the_request_carries_the_midnight_its_slots_count_from():
    clock = Clock(YESTERDAY.replace(hour=23, minute=59))
    manager = make_manager(clock)
    manager.run_cycle()
    records = manager.schedulable()
    assert records
    assert records[0]["anchor"] == YESTERDAY.isoformat()


# --- the request going out -------------------------------------------------------------

def _backend():
    from src.interfaces.optimization_backends.optimization_backend_local_evopt import (
        LocalEVOptBackend,
    )

    backend = LocalEVOptBackend.__new__(LocalEVOptBackend)
    backend.time_zone = BERLIN
    backend.time_frame_base = 900
    return backend


def _at(moment):
    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    return patch(
        "src.interfaces.optimization_backends.optimization_backend_evopt.datetime",
        _Clock,
    )


def test_a_record_from_yesterday_reaches_the_solver_on_todays_hours():
    """01:00 today is slot 100 of yesterday's grid and must still be 01:00 to the solver."""
    mask = [False] * SLOTS
    mask[100] = True
    ambient = [5.0] * SLOTS
    ambient[100] = 11.0
    record = {
        "id": "house", "anchor": YESTERDAY.isoformat(),
        "demand_wh": 3000.0, "max_power_w": 1000.0, "value_eur_per_wh": 0.001,
        "feasible": mask,
        "store": {
            "capacity_wh_per_k": 20000.0, "t_initial": 21.0, "t_min": 21.0,
            "t_target": 21.5, "t_max": 22.5,
            "loss_w_per_k": [50.0] * SLOTS, "ambient_c": ambient,
            "cop": [4.0] * SLOTS, "gain_wh": [0.0] * SLOTS,
        },
    }
    backend = _backend()
    with _at(TODAY.replace(minute=3)):
        configs = backend._managed_load_configs([record])  # pylint: disable=protected-access
    config = configs[0]
    assert config.feasible.index(True) == 4            # four quarter-hours after 00:00
    assert config.store.ambient_c[4] == pytest.approx(11.0)
    assert config.store.ambient_c[0] == pytest.approx(5.0)


def test_slots_a_moved_record_never_described_stay_closed():
    """
    A record from 23:59 moved onto a 00:03 run is a day short at its end. Those slots
    stay closed to the load, as everywhere else a mask runs out, instead of opening
    the whole last day to a load without a window.
    """
    record = {
        "id": "pool", "anchor": YESTERDAY.isoformat(),
        "demand_wh": 3000.0, "max_power_w": 1000.0, "value_eur_per_wh": 0.001,
        "feasible": [True] * SLOTS, "efficiency": [1.0] * SLOTS,
    }
    backend = _backend()
    with _at(TODAY.replace(minute=3)):
        moved = backend._on_run_anchor(record)  # pylint: disable=protected-access
    assert moved["feasible"][: SLOTS - 96] == [True] * (SLOTS - 96)
    assert moved["feasible"][SLOTS - 96:] == [False] * 96


def test_the_response_names_the_runs_midnight():
    backend = _backend()
    with _at(TODAY.replace(minute=3)):
        assert backend._run_anchor() == TODAY          # pylint: disable=protected-access
