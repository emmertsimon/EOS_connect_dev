"""
A heat pump's COP never falls as the air it draws from warms.

A heat-meter fit can come out falling - say 2.74 at 7 C and 2.58 at 15 C - when the
windows mixed in something the air did not cause (a hotter tank in the afternoon,
tapping). Planned on, that slope moves the heating into the cold hours - the wrong way
round. These tests pin the floor: a falling slope comes out flat, the measured level
stays where the windows put it, and neither the configuration nor a persisted estimate
can bring a negative slope back in.
"""

import logging
import math
from datetime import datetime, timedelta, timezone

import pytest

from src.loads.models.calibration import (
    AIR_COEFFICIENT_MIN,
    IDLE_POWER_W,
    ThermalCalibrator,
)
from src.loads.models.thermal_physics import WH_PER_M3_PER_K, cop_at

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)

# A meter suggesting a COP that falls by 0.02 per kelvin of warmer air.
FALLING_NOMINAL = 2.35
FALLING_AIR = -0.009


def _tank(days, mean_c, swing_k, nominal, air, step_minutes=5):
    """
    A store outdoors, heated four times a day by a pump drawing on the same air, so
    the one ambient series drives both the losses and the COP.
    """
    samples = []
    step_h = step_minutes / 60.0
    moment, temp, heat_kwh = T0, 50.0, 0.0
    capacity = WH_PER_M3_PER_K * 0.36
    for _ in range(int(days * 24 / step_h)):
        hour = moment.hour + moment.minute / 60.0
        outdoor = mean_c + swing_k * math.sin(math.pi * (hour - 9) / 12.0)
        running = moment.hour in (3, 4, 9, 10, 13, 14, 20, 21) and temp < 55.0
        power = 1500.0 if running else 10.0
        thermal = power * cop_at(outdoor, nominal, air) if running else 0.0
        samples.append({
            "timestamp": moment,
            "medium_c": round(temp, 2),
            "ambient_c": outdoor,
            "power_w": power,
            "heat_counter_kwh": round(heat_kwh, 2),
        })
        loss = 1.0 * 3.5 * (temp - outdoor) + (400.0 if 7 <= moment.hour < 22 else 0.0)
        temp += (thermal - loss) * step_h / capacity
        heat_kwh += thermal * step_h / 1000.0
        moment += timedelta(minutes=step_minutes)
    return samples


def _tank_calibrator(air=0.0):
    return ThermalCalibrator(
        volume_m3=0.36, surface_m2=3.5, loss_coefficient=1.0,
        cop_nominal=2.9, air_coefficient=air,
    )


def test_a_falling_heat_meter_slope_comes_out_flat():
    calibrator = _tank_calibrator()
    samples = _tank(6, 9.0, 4.0, FALLING_NOMINAL, FALLING_AIR)
    calibrator.observe_series(samples)

    assert calibrator.cop_source == "heat_meter"
    assert calibrator.slope_identified
    assert calibrator.air_coefficient == AIR_COEFFICIENT_MIN
    # Flat, not shifted: the level is the one the meter read across the windows, so the
    # pump is planned at what it actually delivered rather than at the 26 C end of the
    # falling line.
    assert calibrator.cop_nominal == pytest.approx(
        cop_at(9.0, FALLING_NOMINAL, FALLING_AIR), rel=0.05
    )


def test_a_rising_heat_meter_slope_is_left_alone():
    """The floor only takes away what no heat pump does; a normal curve is untouched."""
    calibrator = _tank_calibrator(air=0.007)
    calibrator.observe_series(_tank(6, 9.0, 4.0, 2.9, 0.007))
    assert calibrator.air_coefficient > 0.004


# -- the temperature fit, where there is no heat meter ----------------------------------

VOLUME = 30.0
SURFACE = 32.0
TRUE_K = 25.0


def _store(hours, medium_c, ambient_c, power_w, start, nominal, air):
    samples = []
    step_h = 0.25
    moment, temp = start, medium_c
    for _ in range(int(hours / step_h) + 1):
        samples.append({
            "timestamp": moment,
            "medium_c": temp,
            "ambient_c": ambient_c,
            "power_w": power_w,
            "cover_factor": 1.0,
            "covered": False,
            "solar_w": 0.0,
        })
        loss_w = TRUE_K * SURFACE * (temp - ambient_c)
        thermal_w = (
            power_w * cop_at(ambient_c, nominal, air) if power_w >= IDLE_POWER_W else 0.0
        )
        temp += (thermal_w - loss_w) * step_h / (WH_PER_M3_PER_K * VOLUME)
        moment += timedelta(hours=step_h)
    return samples


def test_a_falling_temperature_fit_slope_comes_out_flat():
    calibrator = ThermalCalibrator(VOLUME, SURFACE, 60.0, 3.0)
    start, temperature = T0, 24.0
    for ambient in (8.0, 14.0, 20.0, 26.0, 30.0):
        for power in (2000.0, 0.0, 2000.0, 0.0):
            samples = _store(4, temperature, ambient, power, start, 4.5, -0.02)
            calibrator.observe_series(samples)
            temperature = samples[-1]["medium_c"]
            start += timedelta(hours=4)
        start += timedelta(hours=8)

    assert calibrator.slope_identified is True
    assert calibrator.air_coefficient == AIR_COEFFICIENT_MIN
    # Re-solved with the slope held, not clipped afterwards: the level lands inside the
    # range the falling curve actually spanned instead of at its 26 C end.
    low = cop_at(30.0, 4.5, -0.02)
    high = cop_at(8.0, 4.5, -0.02)
    assert low * 0.95 <= calibrator.cop_nominal <= high * 1.05


def test_a_negative_configured_slope_is_floored(caplog, monkeypatch):
    from src.loads.models import calibration

    monkeypatch.setattr(calibration, "_WARNED_FALLING_AIR", set())
    with caplog.at_level(logging.WARNING, logger="__main__"):
        calibrator = _tank_calibrator(air=-0.03)
        _tank_calibrator(air=-0.03)
    assert calibrator.configured_air_coefficient == AIR_COEFFICIENT_MIN
    assert calibrator.air_coefficient == AIR_COEFFICIENT_MIN
    # Said once, not on every model rebuild.
    warnings = [r for r in caplog.records if "cop_air_coeff" in r.getMessage()]
    assert len(warnings) == 1


def test_a_persisted_falling_slope_does_not_come_back():
    calibrator = _tank_calibrator()
    calibrator.restore({
        "cop_source": "heat_meter", "cop_nominal": 2.346, "air_coefficient": -0.00894,
    })
    assert calibrator.air_coefficient == AIR_COEFFICIENT_MIN
