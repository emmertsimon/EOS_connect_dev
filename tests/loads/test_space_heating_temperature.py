"""
A house planned by its temperature: the store handed to the solver, and the gate that
follows it.

Planned by energy, a house gets an amount placed in the cheapest hours and says nothing
about when the rooms are warm. Planned by temperature, the model describes the store -
capacity, losses, COP and sun per slot, and a comfort band - and inside that band the
solver's plan decides, not the deadband.
"""

import pytest

from src.loads.presets import TYPE_SPACE_HEATING, apply_defaults

MEASURED_HOUSE = {
    "id": "house",
    "type": TYPE_SPACE_HEATING,
    "living_area_m2": 170,
    "construction": "massive",
    "emitter": "floor",
    "building_form": "detached",
    "insulation": "renovated",
}

HOUSE_ENTRY = dict(
    MEASURED_HOUSE,
    temp_sensor="sensor.rooms",
    power_sensor="sensor.hp_power",
)


def _house_manager(make_manager, installation, rooms=21.2, **overrides):
    installation.sensors.update({"sensor.rooms": rooms, "sensor.hp_power": 0.0})
    installation.temperature = [5.0] * 24 + [9.0] * 24
    manager = make_manager([dict(HOUSE_ENTRY, **overrides)])
    manager.external_scheduler = True
    manager.run_cycle()
    return manager


def test_a_house_is_planned_by_its_temperature_by_default():
    resolved = apply_defaults(MEASURED_HOUSE)
    assert resolved["planning_mode"] == "temperature"
    assert resolved["efficiency_source"] == "cop"


def test_the_store_carries_the_band_and_the_physics(make_manager, installation):
    manager = _house_manager(make_manager, installation)
    store = manager.instance("house").last_demand.store
    assert store["capacity_wh_per_k"] == pytest.approx(21250, rel=0.01)
    assert store["t_min"] == pytest.approx(21.0)
    assert store["t_target"] == pytest.approx(21.5)
    assert store["t_max"] == pytest.approx(22.5)
    assert store["t_initial"] == pytest.approx(21.2)
    assert len(store["cop"]) == len(store["loss_w_per_k"]) == 48
    # Colder slots, lower COP.
    assert store["cop"][0] < store["cop"][30]


def test_the_store_reaches_the_solver_even_without_a_demand(make_manager, installation):
    manager = _house_manager(make_manager, installation, rooms=22.4)
    records = manager.schedulable()
    assert records and records[0]["store"] is not None
    assert records[0]["urgent_wh"] == 0.0


def test_planned_by_energy_there_is_no_store(make_manager, installation):
    manager = _house_manager(make_manager, installation, planning_mode="energy")
    assert manager.instance("house").last_demand.store is None


def test_inside_the_deadband_a_solver_may_still_run_it(make_manager, installation):
    """Banking heat above 21.0 is the solver's call, not the deadband's."""
    manager = _house_manager(make_manager, installation, rooms=21.3)
    item = manager.instance("house")
    assert item.last_demand.detail["active"] is False
    assert item.last_demand.detail["storage_active"] is True
    plan = [0.0] * 48
    plan[6] = 900.0  # 06:00, the slot "now" falls in
    ctx = manager._last_ctx_for["house"]
    release = item.adopt_schedule(plan, ctx)
    assert release["released"] is True


def test_at_the_ceiling_it_stops_whatever_the_plan_says(make_manager, installation):
    manager = _house_manager(make_manager, installation, rooms=22.6)
    item = manager.instance("house")
    plan = [0.0] * 48
    plan[6] = 900.0
    release = item.adopt_schedule(plan, manager._last_ctx_for["house"])
    assert release["released"] is False


def test_without_a_solver_the_deadband_still_decides(make_manager, installation):
    installation.sensors.update({"sensor.rooms": 21.3, "sensor.hp_power": 0.0})
    manager = make_manager([HOUSE_ENTRY])
    manager.run_cycle()
    item = manager.instance("house")
    assert item.last_release["released"] is False


def test_the_projection_may_rise_above_target_when_planned_by_temperature(
    make_manager, installation
):
    manager = _house_manager(make_manager, installation, rooms=21.4)
    item = manager.instance("house")
    ctx = manager._last_ctx_for["house"]
    plan = [0.0] * 48
    for slot in range(6, 14):
        plan[slot] = 3000.0
    projected = item.model.project_medium(ctx, plan, 21.4)
    assert max(value for value in projected if value is not None) > 21.5


def test_the_record_makes_it_into_solver_space(make_manager, installation):
    from datetime import datetime
    from unittest.mock import patch
    from zoneinfo import ZoneInfo

    from src.interfaces.optimization_backends.optimization_backend_local_evopt import (
        LocalEVOptBackend,
    )

    manager = _house_manager(make_manager, installation)
    record = manager.schedulable()[0]
    backend = LocalEVOptBackend.__new__(LocalEVOptBackend)
    backend.time_zone = ZoneInfo("Europe/Berlin")
    backend.time_frame_base = 3600
    moment = datetime(2026, 6, 1, 6, 30, tzinfo=backend.time_zone)

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    with patch(
        "src.interfaces.optimization_backends.optimization_backend_evopt.datetime",
        _Clock,
    ):
        configs = backend._managed_load_configs([record])
    store = configs[0].store
    assert store is not None
    assert store.t_min == pytest.approx(21.0)
    # Rotated: solver slot 0 is 06:00.
    assert store.cop[0] == pytest.approx(record["store"]["cop"][6])


def test_an_unreadable_store_falls_back_to_energy():
    from src.interfaces.optimization_backends.optimization_backend_local_evopt import (
        LocalEVOptBackend,
    )

    backend = LocalEVOptBackend.__new__(LocalEVOptBackend)
    assert backend._solver_store({"capacity_wh_per_k": "lots"}) is None
    assert backend._solver_store(None) is None


def test_a_store_without_a_demand_waits_for_the_solver_it_was_offered_to(
    make_manager, installation
):
    """
    Offered and waited on under the same rule. A warm house wants nothing right now,
    yet it is handed to the solver, which may bank heat later. Judged as idle by the
    handover instead, it would settle its own gate as well - two planners for one heat
    pump, and no notice that the solver never answered.
    """
    installation.sensors.update({"sensor.rooms": 22.4, "sensor.hp_power": 0.0})
    installation.temperature = [24.0] * 48  # a warm spell: no losses to make up
    manager = make_manager([HOUSE_ENTRY])
    manager.external_scheduler = True
    manager.run_cycle()
    item = manager.instance("house")
    assert item.last_demand.total_wh == 0
    assert manager.schedulable(), "the store was not offered"
    assert item.waiting_for_schedule_since is not None, (
        "offered to the solver but not waiting for it"
    )


CURVE = "base:1810, air:-109, store:-880, ref:21"


def _curve_manager(make_manager, installation, rooms=21.2, **overrides):
    settings = dict(rated_power_w=2250, min_power_w=380, power_curve=CURVE,
                    power_curve_learning=False)
    settings.update(overrides)
    return _house_manager(make_manager, installation, rooms=rooms, **settings)


def test_a_heat_pump_on_a_curve_hands_its_power_per_slot_to_the_solver(
    make_manager, installation
):
    """
    The plan switches it; what it draws while on is the curve's - more in the cold
    hours (5 C) than in the mild ones (9 C), less the warmer the house.
    """
    manager = _curve_manager(make_manager, installation)
    record = manager.schedulable()[0]
    ctx = manager._last_ctx_for["house"]
    powers, curve = record["power_w"], record["power_curve"]
    assert len(powers) == ctx.slot_count == len(curve["base_w"])
    assert curve["store_w_per_k"] == pytest.approx(-880)
    assert curve["ref_c"] == pytest.approx(21.0)
    assert (curve["min_w"], curve["max_w"]) == (380, 2250)
    now = ctx.current_slot
    # In the running slot the store is where it is: 21.2 C, 0.2 K above ref.
    assert powers[now] == pytest.approx(curve["base_w"][now] - 880 * 0.2, abs=0.2)
    air = ctx.ambient_temp_c
    cold = [i for i in range(now, ctx.slot_count) if air[i] < 6]
    mild = [i for i in range(now, ctx.slot_count) if air[i] > 8]
    assert cold and mild
    assert min(powers[i] for i in cold) > max(powers[i] for i in mild)
    assert all(380 <= p <= 2250 for p in powers)


def test_the_curve_draw_is_on_the_card(make_manager, installation):
    manager = _curve_manager(make_manager, installation)
    item = manager.instance("house")
    now = manager._last_ctx_for["house"].current_slot
    detail = item.last_demand.detail
    assert detail["power_curve"]["source"] == "configured"
    assert detail["power_curve"]["curve"]["store_w_per_k"] == pytest.approx(-880)
    assert detail["electrical_power_w"] == pytest.approx(item.last_demand.power_w[now], abs=0.1)


def test_a_fixed_heat_pump_hands_over_no_curve(make_manager, installation):
    record = _house_manager(make_manager, installation).schedulable()[0]
    assert record["power_w"] is None and record["power_curve"] is None


def test_a_minimum_power_alone_starts_the_curve_at_the_rating(make_manager, installation):
    """A modulating pump without a curve yet: its rating everywhere until it learns."""
    record = _house_manager(
        make_manager, installation, rated_power_w=2800, min_power_w=300,
    ).schedulable()[0]
    assert set(record["power_w"]) == {2800.0}
    assert record["power_curve"]["store_w_per_k"] == 0


def test_a_broken_power_curve_is_dropped_with_a_warning(make_manager, installation, caplog):
    with caplog.at_level("WARNING"):
        manager = _curve_manager(make_manager, installation, power_curve="base=lots")
    record = manager.schedulable()[0]
    assert "power_curve ignored" in caplog.text
    assert set(record["power_w"]) == {2250.0}


def test_the_modulation_factor_rides_along_with_the_curve(make_manager, installation):
    record = _curve_manager(
        make_manager, installation,
        modulation_cop="400:1.0, 800:1.0, 1200:0.93, 1500:0.81, 1700:0.74",
    ).schedulable()[0]
    table = record["power_curve"]["cop_factor"]
    assert table[0] == [380.0, 1.0]
    assert table[-1][0] == 2250.0 and table[-1][1] < 0.74


def test_a_broken_modulation_curve_is_dropped_with_a_warning(make_manager, installation, caplog):
    with caplog.at_level("WARNING"):
        manager = _curve_manager(make_manager, installation, modulation_cop="400-1.0")
    record = manager.schedulable()[0]
    assert record["power_curve"]["cop_factor"] == []
    assert "modulation_cop ignored" in caplog.text


def test_the_curve_makes_it_into_solver_space(make_manager, installation):
    from datetime import datetime, timedelta
    from unittest.mock import patch
    from zoneinfo import ZoneInfo

    from src.interfaces.optimization_backends.optimization_backend_local_evopt import (
        LocalEVOptBackend,
    )

    record = _curve_manager(make_manager, installation).schedulable()[0]
    backend = LocalEVOptBackend.__new__(LocalEVOptBackend)
    backend.time_zone = ZoneInfo("Europe/Berlin")
    backend.time_frame_base = 3600
    moment = datetime(2026, 6, 1, 6, 30, tzinfo=backend.time_zone)

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    with patch(
        "src.interfaces.optimization_backends.optimization_backend_evopt.datetime",
        _Clock,
    ):
        record["anchor"] = moment.replace(hour=0, minute=0).isoformat()
        load = backend._managed_load_configs([record])[0]
        # A record from yesterday's grid is moved onto today's like every other series.
        stale = dict(record, anchor=(moment.replace(hour=0, minute=0)
                                     - timedelta(days=1)).isoformat())
        moved = backend._managed_load_configs([stale])[0]
    # Rotated: solver slot 0 is 06:00.
    assert load.power_w[0] == pytest.approx(record["power_w"][6])
    assert load.power_curve.base_w[0] == pytest.approx(record["power_curve"]["base_w"][6])
    assert load.power_curve.store_w_per_k == pytest.approx(-880)
    assert load.power_at(0) <= load.max_power_w == 2250
    assert moved.power_w[0] == pytest.approx(record["power_w"][6 + 24])


def test_a_slot_planned_at_the_curve_power_releases(make_manager, installation):
    manager = _curve_manager(make_manager, installation, rooms=21.3, min_runtime_minutes=0)
    item = manager.instance("house")
    ctx = manager._last_ctx_for["house"]
    plan = [0.0] * 48
    plan[ctx.current_slot] = item.last_demand.power_at(ctx.current_slot)
    assert item.adopt_schedule(plan, ctx)["released"] is True
    plan[ctx.current_slot] = 0.2 * item.last_demand.power_at(ctx.current_slot)
    assert item.adopt_schedule(plan, ctx)["released"] is False


# --- what counts as "planned to run" ----------------------------------------------------------

def test_solver_residue_in_the_running_slot_does_not_release(make_manager, installation):
    manager = _house_manager(make_manager, installation, rooms=21.3,
                             min_runtime_minutes=0)
    item = manager.instance("house")
    plan = [0.0] * 48
    plan[6] = 1e-6
    release = item.adopt_schedule(plan, manager._last_ctx_for["house"])
    assert release["released"] is False


def test_residue_does_not_hold_a_running_appliance_on(make_manager, installation):
    """The latch seen live: released once, then held all night by a 0 W plan."""
    manager = _house_manager(make_manager, installation, rooms=21.3,
                             min_runtime_minutes=0)
    item = manager.instance("house")
    ctx = manager._last_ctx_for["house"]
    plan = [0.0] * 48
    plan[6] = 900.0
    assert item.adopt_schedule(plan, ctx)["released"] is True
    plan[6] = 3e-4
    assert item.adopt_schedule(plan, ctx)["released"] is False


def test_a_run_in_the_last_seconds_of_a_slot_still_counts(make_manager, installation):
    import dataclasses
    from datetime import timedelta

    manager = _house_manager(make_manager, installation, rooms=21.3,
                             min_runtime_minutes=0)
    item = manager.instance("house")
    ctx = manager._last_ctx_for["house"]
    slot_end = ctx.anchor + timedelta(seconds=(ctx.current_slot + 1) * ctx.time_frame_base)
    late = dataclasses.replace(ctx, now=slot_end - timedelta(seconds=90))
    plan = [0.0] * 48
    plan[6] = 900.0 * 90 / 3600  # the rating, for the 90 s that are left
    assert item.adopt_schedule(plan, late)["released"] is True


def test_residue_is_not_reported_as_the_next_start(make_manager, installation):
    manager = _house_manager(make_manager, installation, rooms=21.3,
                             min_runtime_minutes=0)
    item = manager.instance("house")
    plan = [0.0] * 48
    plan[10] = 2e-5
    release = item.adopt_schedule(plan, manager._last_ctx_for["house"])
    assert release["next_release_start"] is None
    assert item.last_plan[10] == 0.0
