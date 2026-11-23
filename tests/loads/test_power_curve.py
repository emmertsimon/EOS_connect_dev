"""
The power a modulating heat pump draws while running: the formula, how it is written,
and how the recorded samples refine it.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.loads.power_curve import (
    PowerCurve,
    PowerCurveError,
    PowerCurveLearner,
    curve_from_config,
    factor_table,
    guess_store_c,
    parse_power_curve,
    validate_power_curve_text,
)

# Fitted on a Bosch CS6800iAW over 35 heating runs (Nov 2026).
MEASURED = PowerCurve(base_w=1810, air_w_per_k=-109, store_w_per_k=-880, ref_c=21.0,
                      min_w=380, max_w=2250)
START = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc)


# --- the formula -------------------------------------------------------------------------------

def test_terms_are_read_by_name_and_alias():
    assert parse_power_curve("base:1810, air:-109, store:-880, ref:21") == {
        "base": 1810.0, "air": -109.0, "store": -880.0, "ref": 21.0,
    }
    assert parse_power_curve("p0=1810, outdoor=-109, house=-880") == {
        "base": 1810.0, "air": -109.0, "store": -880.0,
    }


def test_a_decimal_comma_works_with_semicolons():
    assert parse_power_curve("base:1810; air:-109,5; ref:21,5") == {
        "base": 1810.0, "air": -109.5, "ref": 21.5,
    }


def test_empty_means_no_curve():
    assert parse_power_curve("") == {} and parse_power_curve(None) == {}


@pytest.mark.parametrize(
    "text, fragment",
    [("base=lots", "not a term"), ("speed:3", "not a term"), ("base:1, p0:2", "given twice"),
     ("ref:400", "between -50 and 150")],
)
def test_bad_terms_are_refused_with_the_reason(text, fragment):
    with pytest.raises(PowerCurveError, match=fragment):
        parse_power_curve(text)
    assert fragment in validate_power_curve_text(text)


def test_the_draw_follows_air_and_house_within_the_bounds():
    assert MEASURED.at(0, 21.0) == pytest.approx(1810)
    assert MEASURED.at(10, 21.0) == pytest.approx(720)
    assert MEASURED.at(0, 21.6) == pytest.approx(1810 - 528)
    assert MEASURED.at(-10, 20.8) == 2250  # flat out
    assert MEASURED.at(15, 21.6) == 380  # never below its minimum


def test_without_a_base_the_curve_starts_at_the_rating():
    curve = curve_from_config({}, rated_power_w=2800, min_power_w=300, default_ref_c=21.5)
    assert curve.at(-10, 19) == curve.at(20, 23) == 2800
    assert curve.ref_c == 21.5 and curve.min_w == 300


def test_without_a_minimum_running_means_a_tenth_of_the_rating():
    curve = curve_from_config({"base": 100}, rated_power_w=2000, min_power_w=0, default_ref_c=20)
    assert curve.min_w == 200 and curve.at(0, 20) == 200


def test_the_first_guess_settles_back_to_target():
    assert guess_store_c(20.6, 21.5, 0) == pytest.approx(20.6)
    assert guess_store_c(20.6, 21.5, 6) == pytest.approx(21.5 - 0.9 / np.e)
    assert guess_store_c(20.6, 21.5, 48) == pytest.approx(21.5, abs=0.01)


def test_the_modulation_factor_is_tabulated_across_the_range():
    table = factor_table([(400, 1.0), (1700, 0.74)], 380, 2250, steps=4)
    assert [row[0] for row in table] == [380.0, 847.5, 1315.0, 1782.5, 2250.0]
    assert table[0][1] == 1.0 and table[-1][1] < 0.74
    assert factor_table([], 380, 2250) == []


# --- learning it ------------------------------------------------------------------------------

def _runs(curve, airs, houses, noise=0.0, seed=1, minutes=5):
    """Samples of runs at each (air, house): an off sample, then six running ones."""
    rng = np.random.default_rng(seed)
    samples, moment = [], START
    for air, house in zip(airs, houses):
        samples.append({"timestamp": moment, "power_w": 20.0, "medium_c": house,
                        "ambient_c": air})
        for _ in range(6):
            moment += timedelta(minutes=minutes)
            power = curve.at(air, house) + rng.normal(0.0, noise) if noise else curve.at(air, house)
            samples.append({"timestamp": moment, "power_w": float(power),
                            "medium_c": house, "ambient_c": air})
        moment += timedelta(minutes=minutes)
    return samples


def _spread(n=120, seed=3):
    rng = np.random.default_rng(seed)
    return list(rng.uniform(-4, 12, n)), list(rng.uniform(20.8, 21.7, n))


def test_a_house_measured_across_the_weather_is_learned_from_a_flat_start():
    prior = curve_from_config({}, rated_power_w=2250, min_power_w=380, default_ref_c=21.0)
    learner = PowerCurveLearner(prior, idle_power_w=100)
    airs, houses = _spread()
    learner.observe_series(_runs(MEASURED, airs, houses, noise=120.0))
    curve = learner.curve()
    assert learner.source() == "learned"
    assert curve.air_w_per_k == pytest.approx(-109, rel=0.15)
    assert curve.store_w_per_k == pytest.approx(-880, rel=0.2)
    for air, house in ((-2, 21.0), (5, 21.3), (10, 21.5)):
        assert curve.at(air, house) == pytest.approx(MEASURED.at(air, house), abs=120)
    assert learner.state()["fit"]["rmse_w"] < 200


def test_readings_at_the_rating_say_only_at_the_limit():
    """Flat out in the frost is a floor, not a point on the line: the slope survives."""
    prior = curve_from_config({}, rated_power_w=2250, min_power_w=380, default_ref_c=21.0)
    learner = PowerCurveLearner(prior, idle_power_w=100)
    rng = np.random.default_rng(5)
    airs = list(rng.uniform(-8, 12, 150))
    houses = list(rng.uniform(20.8, 21.6, 150))
    learner.observe_series(_runs(MEASURED, airs, houses))
    curve = learner.curve()
    assert curve.air_w_per_k == pytest.approx(-109, rel=0.15)
    assert learner.state()["fit"]["at_limit"] > 0


def test_a_few_runs_barely_move_the_configured_curve():
    learner = PowerCurveLearner(MEASURED, idle_power_w=100)
    learner.observe_series(_runs(MEASURED.with_coefficients(1400, -109, -880), [8.0], [21.2]))
    assert learner.curve().at(8, 21.2) == pytest.approx(MEASURED.at(8, 21.2), abs=250)
    assert learner.curve().at(8, 21.2) < MEASURED.at(8, 21.2)


def test_where_nothing_was_measured_the_configuration_holds():
    """A week of mild weather says little about a frost."""
    learner = PowerCurveLearner(MEASURED, idle_power_w=100)
    rng = np.random.default_rng(7)
    airs, houses = list(rng.uniform(8, 12, 80)), list(rng.uniform(20.9, 21.5, 80))
    learner.observe_series(_runs(MEASURED, airs, houses, noise=60.0))
    assert learner.curve().at(-5, 21.0) == pytest.approx(MEASURED.at(-5, 21.0), abs=200)


def test_the_first_minutes_of_a_run_are_not_learned():
    learner = PowerCurveLearner(MEASURED, idle_power_w=100)
    learned = [
        learner.observe({"timestamp": START + timedelta(minutes=m), "power_w": 1000.0,
                         "medium_c": 21.0, "ambient_c": 5.0})
        for m in (0, 5, 10, 15)
    ]
    assert learned == [False, False, True, True]


def test_a_gap_starts_a_new_run():
    learner = PowerCurveLearner(MEASURED, idle_power_w=100)
    for m in (0, 5, 10):
        learner.observe({"timestamp": START + timedelta(minutes=m), "power_w": 900.0,
                         "medium_c": 21.0, "ambient_c": 5.0})
    assert not learner.observe({"timestamp": START + timedelta(minutes=60), "power_w": 900.0,
                                "medium_c": 21.0, "ambient_c": 5.0})


def test_standby_and_implausible_readings_are_skipped():
    learner = PowerCurveLearner(MEASURED, idle_power_w=100)
    for m in range(0, 30, 5):
        learner.observe({"timestamp": START + timedelta(minutes=m), "power_w": 1000.0,
                         "medium_c": 21.0, "ambient_c": 5.0})
    count = len(learner.samples)
    assert not learner.observe({"timestamp": START + timedelta(minutes=30), "power_w": 9000.0,
                                "medium_c": 21.0, "ambient_c": 5.0})
    assert not learner.observe({"timestamp": START + timedelta(minutes=35), "power_w": 40.0,
                                "medium_c": 21.0, "ambient_c": 5.0})
    assert len(learner.samples) == count


def test_the_heat_pumps_own_air_comes_first():
    learner = PowerCurveLearner(MEASURED, idle_power_w=100)
    for m in (0, 5, 10):
        learner.observe({"timestamp": START + timedelta(minutes=m), "power_w": 1000.0,
                         "medium_c": 21.0, "ambient_c": 18.0, "cop_air_c": 2.0})
    assert learner.samples[-1][0] == 2.0


def test_with_learning_off_the_configuration_is_used_as_it_is():
    learner = PowerCurveLearner(MEASURED, idle_power_w=100, learn=False)
    airs, houses = _spread(40)
    learner.observe_series(_runs(MEASURED.with_coefficients(500, 0, 0), airs, houses))
    assert learner.curve() is MEASURED and learner.source() == "configured"


def test_a_new_configuration_keeps_the_samples():
    learner = PowerCurveLearner(MEASURED, idle_power_w=100)
    airs, houses = _spread(40)
    learner.observe_series(_runs(MEASURED, airs, houses))
    learner.set_prior(MEASURED.with_coefficients(2000, 0, 0))
    assert len(learner.samples) > 0
    assert learner.curve().air_w_per_k < -50  # still the data's slope, not the new prior's 0


# --- in the model --------------------------------------------------------------------------------

def test_the_model_learns_from_the_replayed_history():
    from src.loads.models.thermal import ThermalStorageModel

    model = ThermalStorageModel("house", {
        "id": "house", "volume_m3": 100, "surface_m2": 170, "rated_power_w": 2250,
        "min_power_w": 380, "target_temp": 21.0,
    })
    airs, houses = _spread()
    model.observe_history(_runs(MEASURED, airs, houses, noise=80.0))
    state = model.status()["power_curve"]
    assert state["source"] == "learned" and state["samples"] > 300
    assert state["curve"]["air_w_per_k"] == pytest.approx(-109, rel=0.2)


def test_an_appliance_at_its_rating_has_no_curve():
    from src.loads.models.thermal import ThermalStorageModel

    model = ThermalStorageModel("pool", {"id": "pool", "volume_m3": 40, "rated_power_w": 2000})
    assert model.power_curve() is None and model.status()["power_curve"] is None
