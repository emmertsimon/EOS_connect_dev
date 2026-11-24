"""
A house learns its capacity, loss and free heat from how well it forecast.

The windows the calibrator fits cannot learn a house with floor heating - the screed
holds a run's heat back for hours - and the capacity was never learned at all. These
tests run a simulated house whose numbers are known, record it the way the manager
does, and check that the fit finds the house, that it is only adopted when it forecasts
the held-back days better, and that the plan then uses it.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.loads.models import house_fit
from src.loads.models.calibration import ThermalCalibrator
from src.loads.models.thermal_physics import WH_PER_M3_PER_K
from src.loads.presets import TYPE_SPACE_HEATING, apply_defaults, build_model

TRUE_HOUSE = {
    "capacity_wh_per_k": 36000.0,
    "loss_w_per_k": 100.0,
    "solar_gain": 1.0,
    "internal_gain_w": 150.0,
}
# What the building table guesses for the measured 170 m2 house.
CONFIGURED = {
    "capacity_wh_per_k": 21250.0,
    "loss_w_per_k": 51.0,
    "solar_gain": 0.0,
    "internal_gain_w": 0.0,
}
START = datetime(2026, 11, 10, tzinfo=timezone.utc)


def simulate(days=10, house=TRUE_HOUSE, step_min=5, start_c=21.2):
    """
    A one-store house run on a thermostat with night and noon runs, cold nights, sunny
    middays - recorded every five minutes with a heat meter and a PV meter, like the
    manager records it.
    """
    samples = []
    temperature = start_c
    heat_kwh = 100.0
    pv_kwh = 500.0
    hours = step_min / 60.0
    steps = int(days * 24 * 60 / step_min)
    for index in range(steps):
        moment = START + timedelta(minutes=step_min * index)
        hour = (moment.hour + moment.minute / 60.0)
        day = index * step_min / 1440.0
        air = 2.0 + 4.0 * math.sin((hour - 9.0) / 24.0 * 2 * math.pi) + 3.0 * math.sin(day)
        sun = max(0.0, math.sin((hour - 7.0) / 10.0 * math.pi)) * 3000.0 if 7 <= hour <= 17 else 0.0
        sun *= 0.5 + 0.5 * abs(math.sin(day * 1.7))
        running = (3 <= hour < 6) or (12 <= hour < 14.5) or temperature < 20.8
        heat_w = 4500.0 if running else 0.0
        samples.append({
            "timestamp": moment,
            "medium_c": round(temperature, 2),
            "ambient_c": round(air, 1),
            "power_w": 1300.0 if running else 0.0,
            "heat_counter_kwh": round(heat_kwh, 2),
            "pv_counter_kwh": round(pv_kwh, 3),
            "solar_w": sun,
        })
        net = (heat_w + house["solar_gain"] * sun + house["internal_gain_w"]
               - house["loss_w_per_k"] * (temperature - air))
        temperature += hours * net / house["capacity_wh_per_k"]
        heat_kwh += heat_w * hours / 1000.0
        pv_kwh += sun * hours / 1000.0
    return samples


def test_the_fit_finds_the_house_and_beats_the_guess():
    result = house_fit.fit_house(simulate(), CONFIGURED)
    assert result["accepted"] is True
    # Loss, sun and the house's own heat trade off against each other a little - the
    # forecast is what has to be right, and it is, to a few hundredths of a kelvin.
    assert result["capacity_wh_per_k"] == pytest.approx(36000, rel=0.2)
    assert result["loss_w_per_k"] == pytest.approx(100, rel=0.2)
    assert result["holdout_rms_k"] < 0.05
    assert result["holdout_rms_k"] < 0.3 * result["holdout_rms_in_use_k"]


def test_a_short_history_is_not_fitted():
    result = house_fit.fit_house(simulate(days=3), CONFIGURED)
    assert result["accepted"] is False
    assert result["reason"] == "not enough history yet"


def test_a_house_without_heat_is_not_fitted():
    samples = simulate()
    for sample in samples:
        sample["heat_counter_kwh"] = 100.0
    result = house_fit.fit_house(samples, CONFIGURED)
    assert result["accepted"] is False


def test_values_that_already_forecast_well_are_kept():
    """A new set has to forecast the held-back days better, not merely differently."""
    result = house_fit.fit_house(simulate(), CONFIGURED, current=TRUE_HOUSE)
    assert result["accepted"] is False
    assert result["capacity_wh_per_k"] == pytest.approx(36000, rel=1e-3)
    assert "better" in result["reason"]


def test_the_fit_is_quick():
    """Twice a day in the poll thread: it must not be noticeable."""
    import time
    samples = simulate(days=14)
    began = time.perf_counter()
    house_fit.fit_house(samples, CONFIGURED)
    assert time.perf_counter() - began < 2.0


def test_gaps_and_a_counter_reset_leave_slots_unknown_rather_than_wrong():
    samples = simulate()
    del samples[1000:1012]          # an hour missing
    for sample in samples[2000:]:   # the heat meter was replaced
        sample["heat_counter_kwh"] -= 90.0
    result = house_fit.fit_house(samples, CONFIGURED)
    assert result["accepted"] is True
    assert result["loss_w_per_k"] == pytest.approx(100, rel=0.2)


def _calibrator(**options):
    return ThermalCalibrator(
        volume_m3=CONFIGURED["capacity_wh_per_k"] / WH_PER_M3_PER_K,
        surface_m2=408.0,
        loss_coefficient=0.125,
        cop_nominal=5.5,
        air_coefficient=0.012,
        loss_coefficient_min=0.02,
        loss_coefficient_max=5.0,
        learn_cop=False,
        min_window_change_k=0.2,
        **options,
    )


def test_the_calibrator_adopts_a_learned_house_and_persists_it():
    calibrator = _calibrator(learn_capacity=True)
    calibrator.observe_series(simulate())
    assert calibrator.house is not None
    assert calibrator.capacity_wh_per_k() == pytest.approx(36000, rel=0.2)
    assert calibrator.effective_loss_coefficient() * 408.0 == pytest.approx(100, rel=0.2)
    state = calibrator.state()
    assert state["capacity_source"] == "house_fit"
    assert state["house_fit"]["accepted"] is True

    restored = _calibrator(learn_capacity=True)
    restored.restore(state)
    assert restored.capacity_wh_per_k() == pytest.approx(calibrator.capacity_wh_per_k())
    assert restored.internal_gain_w() == pytest.approx(calibrator.internal_gain_w())


def test_a_restored_house_outside_the_bounds_is_ignored():
    calibrator = _calibrator(learn_capacity=True)
    calibrator.restore({"house": dict(TRUE_HOUSE, capacity_wh_per_k=500000.0)})
    assert calibrator.house is None
    assert calibrator.capacity_wh_per_k() == pytest.approx(21250, rel=0.01)


def test_a_store_that_does_not_learn_its_capacity_is_unchanged():
    calibrator = _calibrator()
    calibrator.observe_series(simulate())
    assert calibrator.house is None
    assert calibrator.capacity_wh_per_k() == pytest.approx(21250, rel=0.01)
    assert calibrator.effective_loss_coefficient() == calibrator.loss_coefficient
    assert calibrator.internal_gain_w() == 0.0
    assert "capacity_source" not in calibrator.state()


def test_the_live_path_refits_twice_a_day_not_every_sample():
    calibrator = _calibrator(learn_capacity=True)
    samples = simulate(days=8)
    calibrator.observe_series(samples[:-288])
    first = calibrator.house_fit["fitted_at"]
    for sample in samples[-288:-200]:          # seven hours: not due
        calibrator.observe(sample)
    assert calibrator.house_fit["fitted_at"] == first
    for sample in samples[-200:]:              # past twelve hours: refitted once
        calibrator.observe(sample)
    assert calibrator.house_fit["fitted_at"] != first


def test_a_space_heating_store_learns_its_house_by_default():
    entry = apply_defaults({
        "id": "house", "type": TYPE_SPACE_HEATING, "living_area_m2": 170,
        "construction": "massive", "emitter": "floor", "building_form": "detached",
        "insulation": "renovated",
    })
    model = build_model(entry)
    assert model.calibrator.learn_capacity is True


def test_the_plan_uses_the_learned_house(make_manager, installation):
    installation.sensors.update({"sensor.rooms": 21.2, "sensor.hp_power": 0.0})
    installation.temperature = [5.0] * 24 + [9.0] * 24
    manager = make_manager([{
        "id": "house", "type": TYPE_SPACE_HEATING, "living_area_m2": 170,
        "construction": "massive", "emitter": "floor", "building_form": "detached",
        "insulation": "renovated", "temp_sensor": "sensor.rooms",
        "power_sensor": "sensor.hp_power",
    }])
    manager.external_scheduler = True
    manager.run_cycle()
    item = manager.instance("house")
    calibrator = item.model.calibrator
    surface = item.model.surface_m2
    before = item.last_demand.detail["standing_losses_wh_thermal"]
    assert calibrator.solar_gain == 0.0  # the test installation has learned no sun

    calibrator.house = dict(TRUE_HOUSE)
    manager.run_cycle()
    store = item.last_demand.store
    assert store["capacity_wh_per_k"] == pytest.approx(36000)
    assert store["loss_w_per_k"][0] == pytest.approx(100.0, rel=1e-6)
    hours = manager._last_ctx_for["house"].hours_per_slot()
    # The house's own heat in every slot, the sun on top where it shines.
    assert min(store["gain_wh"]) == pytest.approx(150.0 * hours)

    # Twice the configured loss and nothing for free: twice the standing losses.
    configured = calibrator.configured_loss_coefficient * surface
    calibrator.house = dict(
        TRUE_HOUSE, loss_w_per_k=2 * configured, solar_gain=0.0, internal_gain_w=0.0
    )
    manager.run_cycle()
    after = item.last_demand.detail["standing_losses_wh_thermal"]
    assert after == pytest.approx(2 * before, rel=0.02)
