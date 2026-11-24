"""
How far a heat meter may move the COP slope: by the spread of the air, not the count.

A heat meter measures the COP level directly, but the slope - how fast the COP falls
as the air cools - is only as good as the range of air temperatures behind it. With a
prior worth a fixed count of windows, a few kelvin of autumn could fit a slope twice
the configured one and extrapolate it to a frost far below what the meter reads there.
These tests pin the rule: against a narrow spread the configured slope keeps most of
its say, against a winter's spread the meter takes over.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.loads.models.calibration import ThermalCalibrator
from src.loads.models.thermal_physics import WH_PER_M3_PER_K, cop_at

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
# A configured curve: 3.5 at 0 C, 4.15 at 10 C.
PRIOR_NOMINAL = 5.19
PRIOR_AIR = 0.0125
# What a narrow autumn's windows might suggest: 2.6 at 0 C, 4.3 at 10 C.
STEEP_NOMINAL = 7.0
STEEP_AIR = 0.024


def _simulate(days, mean_c, swing_k, nominal, air, step_minutes=5):
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


def _calibrator():
    return ThermalCalibrator(
        volume_m3=0.36, surface_m2=3.5, loss_coefficient=1.0,
        cop_nominal=PRIOR_NOMINAL, air_coefficient=PRIOR_AIR,
    )


def _slope(calibrator):
    return calibrator.cop_nominal * calibrator.air_coefficient


def test_a_narrow_autumn_does_not_overturn_the_configured_slope():
    calibrator = _calibrator()
    calibrator.observe_series(_simulate(5, 10.0, 2.0, STEEP_NOMINAL, STEEP_AIR))
    assert calibrator.cop_source == "heat_meter"
    assert calibrator.slope_identified
    # The level is measured, so it is right where the windows were.
    assert cop_at(10.0, calibrator.cop_nominal, calibrator.air_coefficient) == (
        pytest.approx(cop_at(10.0, STEEP_NOMINAL, STEEP_AIR), rel=0.05)
    )
    # The slope moved towards the windows but kept nearer the configured one, so a
    # frost is not planned at 2.6.
    prior_slope = PRIOR_NOMINAL * PRIOR_AIR
    steep_slope = STEEP_NOMINAL * STEEP_AIR
    assert prior_slope < _slope(calibrator) < (prior_slope + steep_slope) / 2
    assert cop_at(0.0, calibrator.cop_nominal, calibrator.air_coefficient) > 3.0


def test_a_winters_spread_lets_the_meter_set_the_slope():
    calibrator = _calibrator()
    calibrator.observe_series(_simulate(10, 3.0, 9.0, STEEP_NOMINAL, STEEP_AIR))
    assert calibrator.cop_source == "heat_meter"
    for air in (-5.0, 0.0, 10.0):
        assert cop_at(air, calibrator.cop_nominal, calibrator.air_coefficient) == (
            pytest.approx(cop_at(air, STEEP_NOMINAL, STEEP_AIR), rel=0.06)
        )


def test_the_configured_slope_survives_where_it_was_right():
    """A prior that matches the pump is not pulled anywhere by a narrow spread."""
    calibrator = _calibrator()
    calibrator.observe_series(_simulate(5, 10.0, 2.0, PRIOR_NOMINAL, PRIOR_AIR))
    for air in (0.0, 10.0):
        assert cop_at(air, calibrator.cop_nominal, calibrator.air_coefficient) == (
            pytest.approx(cop_at(air, PRIOR_NOMINAL, PRIOR_AIR), rel=0.03)
        )
