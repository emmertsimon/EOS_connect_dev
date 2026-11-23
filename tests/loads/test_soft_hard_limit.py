"""
A hot water tank's soft and hard limit, planned by EOS rather than by automations.

The price limit keeps the tank from being heated at 45 ct; on a day when nothing is
cheaper it also kept it from being heated at all, and two Home Assistant automations
stood in: below 42 C pick the cheapest hour of the midday window, below 32 C charge at
once. These tests pin the same behaviour inside EOS: below the soft limit the charge
lands in the next allowed window whatever the price, in its cheapest slot; below the
hard limit it starts now, window or not.
"""

from datetime import datetime

import pytest

from tests.loads.conftest import BERLIN

from src.loads.models.thermal import REASON_FROST, REASON_SOFT_LIMIT
from src.loads.presets import TYPE_HOT_WATER_TANK

# 06:30 in the fixtures; hourly slots from midnight.
NOW_SLOT = 6
DEAR = 0.00045

TANK = {
    "id": "ww",
    "type": TYPE_HOT_WATER_TANK,
    "temp_sensor": "sensor.ww",
    "power_sensor": "sensor.ww_power",
    "target_temp": 52.0,
    "deadband_k": 7.0,
    "volume_m3": 0.36,
    "rated_power_w": 2900.0,
    "cop_nominal": 2.7,
    "cop_air_coeff": 0.0,
    "window_start": 10,
    "window_end": 16,
    "max_price_ct_kwh": 38,
    "min_runtime_minutes": 30,
}
SOFT = dict(TANK, soft_limit_temp_c=42.0)
SOFT_AND_HARD = dict(SOFT, frost_protection_temp_c=32.0)


def _run(make_manager, installation, entry, tank_c, prices=None, external=False):
    installation.sensors.update({"sensor.ww": tank_c, "sensor.ww_power": 0.0})
    installation.prices = prices or [DEAR] * installation.slot_count
    manager = make_manager([entry])
    manager.external_scheduler = external
    manager.run_cycle()
    return manager, manager.instance("ww")


def _used(plan):
    return [index for index, value in enumerate(plan) if value > 0]


def test_without_a_soft_limit_a_dear_day_heats_nothing(make_manager, installation):
    """The situation of 23.11.: 42-52 ct all day against a 38 ct limit."""
    _, item = _run(make_manager, installation, TANK, 40.0)
    assert sum(item.last_plan) == 0
    assert item.last_demand.must_wh == 0


def test_below_the_soft_limit_the_charge_lands_in_todays_window(
    make_manager, installation
):
    prices = [DEAR] * 48
    prices[13] = DEAR - 0.00003
    _, item = _run(make_manager, installation, SOFT, 40.0, prices)

    demand = item.last_demand
    assert demand.reason == REASON_SOFT_LIMIT
    assert demand.must_wh > 0
    assert [i for i, on in enumerate(demand.must_slots) if on] == list(range(10, 16))
    assert demand.detail["must_by"] == datetime(2026, 6, 1, 16, 0, tzinfo=BERLIN).isoformat()
    # The cheapest hour of the window, past the price limit - and only the charge.
    assert _used(item.last_plan) == [13]
    assert sum(item.last_plan) == pytest.approx(demand.must_wh, rel=0.01)
    # 06:30 is outside the window: nothing runs yet.
    assert item.last_release["released"] is False


def test_above_the_soft_limit_the_price_limit_still_rules(make_manager, installation):
    _, item = _run(make_manager, installation, SOFT, 43.0)
    assert item.last_demand.must_wh == 0
    assert sum(item.last_plan) == 0


def test_the_soft_limit_hours_pull_the_deadline_in(make_manager, installation):
    entry = dict(SOFT, soft_limit_hours=3)
    _, item = _run(make_manager, installation, entry, 40.0)
    slots = [i for i, on in enumerate(item.last_demand.must_slots) if on]
    # Three hours from 06:30 end before the window opens, so the charge's own length
    # from the window's start is what is left.
    assert slots[0] == 10
    assert slots[-1] < 15


def test_below_the_hard_limit_the_charge_starts_now_outside_the_window(
    make_manager, installation
):
    _, item = _run(make_manager, installation, SOFT_AND_HARD, 30.0)
    demand = item.last_demand

    assert demand.reason == REASON_FROST
    assert demand.urgent is True
    assert demand.must_slots[NOW_SLOT] is True
    assert demand.feasible[NOW_SLOT] is True
    assert item.last_release["released"] is True
    assert item.last_plan[NOW_SLOT] > 0


def test_the_optimizer_is_handed_the_charge_and_not_the_whole_horizon(
    make_manager, installation
):
    manager, item = _run(
        make_manager, installation, SOFT_AND_HARD, 30.0, external=True
    )
    record = next(r for r in manager.schedulable() if r["id"] == "ww")
    assert record["must_wh"] == pytest.approx(item.last_demand.must_wh, abs=0.1)
    assert record["must_wh"] < record["demand_wh"]
    assert record["must_slots"][NOW_SLOT] is True
    assert len(record["must_slots"]) == len(record["feasible"])
    # The old hard floor of the whole demand is gone.
    assert record["urgent_wh"] == 0.0


def test_a_tank_without_limits_offers_no_charge(make_manager, installation):
    manager, _ = _run(make_manager, installation, TANK, 44.0, external=True)
    for record in manager.schedulable():
        assert record["must_wh"] == 0
        assert record["must_slots"] == []
