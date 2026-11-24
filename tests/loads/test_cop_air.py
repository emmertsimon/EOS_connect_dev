"""
A tank indoors, a heat pump outdoors: ``cop_air_sensor``.

A hot water tank loses its heat to the room it stands in, while an air-source heat pump
takes the heat it puts in from the outdoor air. One ambient series cannot be both. These
tests pin the split: the COP follows the outdoor forecast matched to the heat pump's own
sensor, the losses stay on the surroundings, and a heat meter is fitted against the air
the heat actually came from.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.loads.models.calibration import ThermalCalibrator
from src.loads.models.thermal_physics import WH_PER_M3_PER_K, cop_at

TANK = {
    "id": "dhw",
    "type": "hot_water_tank",
    "temp_sensor": "sensor.dhw",
    "power_sensor": "sensor.dhw_power",
    "rated_power_w": 2900,
    "target_temp": 52,
    "volume_m3": 0.36,
    "surface_m2": 3.5,
    "cop_nominal": 3.6,
    "cop_air_coeff": 0.0125,
}
TANK_WITH_AIR = dict(TANK, cop_air_sensor="sensor.hp_outdoor")

# A November day: frosty morning, mild afternoon.
DAY = [0.0] * 6 + [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0, 10.0, 9.0, 7.0, 5.0, 3.0] + [1.0] * 6


def _ctx(manager, readings):
    """The per-instance context the cycle would build."""
    item = manager.instance("dhw")
    base = manager._context()  # pylint: disable=protected-access
    original = manager._read_sensors  # pylint: disable=protected-access
    manager._read_sensors = lambda _item: dict(readings)  # pylint: disable=protected-access
    try:
        return item, manager._context_for(item, base)  # pylint: disable=protected-access
    finally:
        manager._read_sensors = original  # pylint: disable=protected-access


# --- where the forecast is needed ---------------------------------------------------------


def test_an_indoor_tank_does_not_ask_for_the_forecast(make_manager):
    assert make_manager([TANK]).needs_outdoor_temperature() is False


def test_naming_the_heat_pumps_air_asks_for_it(make_manager):
    assert make_manager([TANK_WITH_AIR]).needs_outdoor_temperature() is True


# --- the series -----------------------------------------------------------------------------


def test_without_the_sensor_nothing_changes(make_manager, installation):
    installation.temperature = DAY + DAY
    manager = make_manager([TANK])
    item, ctx = _ctx(manager, {"temp_sensor": 50.0})
    assert ctx.cop_air_temp_c == []
    cops = item.model.cop_series(ctx)
    assert max(cops) == pytest.approx(min(cops)), "the room air is flat, so is the COP"


def test_the_cop_follows_the_outdoor_forecast(make_manager, installation):
    installation.temperature = DAY + DAY
    manager = make_manager([TANK_WITH_AIR])
    item, ctx = _ctx(manager, {"temp_sensor": 50.0, "cop_air_sensor": None})

    assert ctx.cop_air_temp_c[13] == pytest.approx(10.0)
    cops = item.model.cop_series(ctx)
    assert cops[13] == pytest.approx(cop_at(10.0, 3.6, 0.0125))
    assert cops[13] > cops[3], "a mild afternoon is cheaper hot water than a frosty night"


def test_the_losses_stay_on_the_surroundings(make_manager, installation):
    """Outdoor air for the losses would put half again on top in winter."""
    installation.temperature = DAY + DAY
    plain = make_manager([TANK])
    with_air = make_manager([TANK_WITH_AIR])
    readings = {"temp_sensor": 50.0, "cop_air_sensor": 4.0}

    item_a, ctx_a = _ctx(plain, readings)
    item_b, ctx_b = _ctx(with_air, readings)
    assert ctx_b.ambient_temp_c == ctx_a.ambient_temp_c
    detail_a = item_a.model.demand(ctx_a).detail
    detail_b = item_b.model.demand(ctx_b).detail
    assert detail_b["standing_losses_wh_thermal"] == pytest.approx(
        detail_a["standing_losses_wh_thermal"]
    )
    assert detail_b["cop_air_measured_c"] == pytest.approx(4.0)
    assert detail_a["cop_air_now_c"] is None


def test_the_reading_carries_where_there_is_no_forecast(make_manager, installation):
    installation.temperature = []
    manager = make_manager([TANK_WITH_AIR])
    _, ctx = _ctx(manager, {"temp_sensor": 50.0, "cop_air_sensor": "6.5 °C"})
    assert ctx.cop_air_temp_c == [6.5] * ctx.slot_count


def test_neither_falls_back_to_the_surroundings(make_manager, installation):
    installation.temperature = []
    manager = make_manager([TANK_WITH_AIR])
    item, ctx = _ctx(manager, {"temp_sensor": 50.0, "cop_air_sensor": "unavailable"})
    assert ctx.cop_air_temp_c == []
    cops = item.model.cop_series(ctx)
    assert max(cops) == pytest.approx(min(cops))


def test_the_forecast_is_matched_to_the_heat_pumps_sensor(make_manager, installation):
    """The same per-hour correction an outdoor load's ambient gets, kept apart from it."""
    installation.temperature = [9.5] * 48
    manager = make_manager([TANK_WITH_AIR])
    for _ in range(20):
        _, ctx = _ctx(manager, {"temp_sensor": 50.0, "cop_air_sensor": 7.1})
    assert ctx.cop_air_temp_c[ctx.current_slot] == pytest.approx(7.1, abs=0.05)
    assert ctx.cop_air_forecast_c == pytest.approx(9.5)
    # The tank's own ambient learned nothing from it.
    assert manager.ambient_bias_state("dhw") is None


def test_an_air_limit_applies_to_the_heat_pumps_air(make_manager, installation):
    installation.temperature = DAY + DAY
    manager = make_manager([dict(TANK_WITH_AIR, min_ambient_temp_c=2.0)])
    item, ctx = _ctx(manager, {"temp_sensor": 50.0})
    mask, _ = item.model.feasibility(ctx)
    assert mask[3] is False, "0 C outside is below the limit"
    assert mask[13] is True


# --- samples and restart --------------------------------------------------------------------


def test_samples_carry_the_heat_pumps_air(make_manager, installation):
    installation.temperature = DAY + DAY
    manager = make_manager([TANK_WITH_AIR])
    installation.sensors.update({
        "sensor.dhw": 50.0, "sensor.dhw_power": 0.0, "sensor.hp_outdoor": 4.0,
    })
    seen = []
    manager.instance("dhw").observe = lambda sample: seen.append(sample) or False
    manager.run_cycle()
    assert seen, "the cycle records a sample"
    assert seen[-1]["cop_air_measured_c"] == pytest.approx(4.0)
    assert seen[-1]["cop_air_forecast_c"] is not None
    assert seen[-1]["cop_air_c"] is not None


def test_the_offsets_are_relearned_from_history(make_manager):
    manager = make_manager([TANK_WITH_AIR])
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    samples = [
        {
            "timestamp": start + timedelta(hours=hour),
            "cop_air_measured_c": 5.0,
            "cop_air_forecast_c": 8.0,
        }
        for hour in range(24 * 6)
    ]
    manager._replay_ambient_bias(  # pylint: disable=protected-access
        manager.instance("dhw"), samples
    )
    bias = manager._ambient_bias["dhw:cop_air"]  # pylint: disable=protected-access
    assert bias.offset(12) == pytest.approx(-3.0, abs=0.3)
    assert "dhw" not in manager._ambient_bias  # pylint: disable=protected-access


# --- the heat meter ------------------------------------------------------------------------

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
TRUE_NOMINAL = 3.6
TRUE_AIR = 0.0125
ROOM_C = 18.0


def _outdoor(moment):
    hour = moment.hour + moment.minute / 60.0
    return 4.0 + 6.0 * math.sin(math.pi * (hour - 9) / 12.0)


def _simulate_tank(days, with_air=True, step_minutes=5):
    """A tank in an 18 C room, heated twice a day by a pump on outdoor air."""
    samples = []
    step_h = step_minutes / 60.0
    moment, temp, heat_kwh = T0, 50.0, 0.0
    capacity = WH_PER_M3_PER_K * 0.36
    for _ in range(int(days * 24 / step_h)):
        running = moment.hour in (3, 4, 13, 14) and temp < 55.0
        power = 1500.0 if running else 10.0
        air = _outdoor(moment)
        thermal = power * cop_at(air, TRUE_NOMINAL, TRUE_AIR) if running else 0.0
        samples.append({
            "timestamp": moment,
            "medium_c": round(temp, 2),
            "ambient_c": ROOM_C,
            "power_w": power,
            "heat_counter_kwh": round(heat_kwh, 2),
            "cop_air_c": air if with_air else None,
            "cop_air_measured_c": air if with_air else None,
        })
        loss = 1.0 * 3.5 * (temp - ROOM_C) + (300.0 if 7 <= moment.hour < 22 else 0.0)
        temp += (thermal - loss) * step_h / capacity
        heat_kwh += thermal * step_h / 1000.0
        moment += timedelta(minutes=step_minutes)
    return samples


def _calibrator():
    return ThermalCalibrator(
        volume_m3=0.36, surface_m2=3.5, loss_coefficient=1.0,
        cop_nominal=3.0, air_coefficient=0.01,
    )


def test_a_heat_meter_is_fitted_against_the_heat_pumps_air():
    calibrator = _calibrator()
    calibrator.observe_series(_simulate_tank(days=6))
    assert calibrator.cop_source == "heat_meter"
    for air in (0.0, 9.0):
        assert cop_at(air, calibrator.cop_nominal, calibrator.air_coefficient) == (
            pytest.approx(cop_at(air, TRUE_NOMINAL, TRUE_AIR), rel=0.05)
        )


def test_against_the_room_alone_the_level_is_right_but_flat():
    """What an indoor tank got before: one air temperature, so no slope to see."""
    calibrator = _calibrator()
    calibrator.observe_series(_simulate_tank(days=6, with_air=False))
    assert calibrator.cop_source == "heat_meter"
    assert not calibrator.slope_identified
    level = cop_at(ROOM_C, calibrator.cop_nominal, calibrator.air_coefficient)
    true_mean = cop_at(4.0, TRUE_NOMINAL, TRUE_AIR)
    assert level == pytest.approx(true_mean, rel=0.08)
