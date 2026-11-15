"""
A modulating appliance's COP across its power range, as the user writes it and as the
solver reads it.
"""

import pytest

from src.loads.modulation import (
    ModulationError,
    factor_at,
    heat_at,
    heat_lines,
    parse_modulation,
    validate_modulation_text,
)

MEASURED = "400:1.0, 800:1.0, 1200:0.93, 1500:0.81, 1700:0.74"


def test_points_are_read_and_sorted():
    assert parse_modulation("800:1, 400:1.0, 1.2kW:0.93") == [
        (400.0, 1.0), (800.0, 1.0), (1200.0, 0.93),
    ]


def test_a_decimal_comma_works_with_semicolons():
    assert parse_modulation("400:1,0; 1200:0,93") == [(400.0, 1.0), (1200.0, 0.93)]


def test_empty_means_no_curve():
    assert parse_modulation("") == [] and parse_modulation(None) == []


@pytest.mark.parametrize(
    "text, fragment",
    [("400-1.0", "not a point"), ("0:1.0", "above 0 W"), ("400:9", "between 0.05 and 3"),
     ("400:1, 400:0.9", "given twice")],
)
def test_bad_points_are_refused_with_the_reason(text, fragment):
    with pytest.raises(ModulationError, match=fragment):
        parse_modulation(text)
    assert fragment in validate_modulation_text(text)


def test_the_factor_is_linear_between_points_and_keeps_falling_beyond_them():
    points = parse_modulation(MEASURED)
    assert factor_at(points, 300) == 1.0
    assert factor_at(points, 1000) == pytest.approx(0.965)
    # Beyond 1700 W the heat grows at the last stretch's rate (43 W per 200 W).
    assert heat_at(points, 2800) == pytest.approx(1258 + 0.215 * 1100)
    assert factor_at(points, 2800) < 0.74


def test_a_concave_curve_is_met_at_its_points_and_never_overrated():
    points = parse_modulation(MEASURED)
    lines = heat_lines(points, 300, 2800)
    for power in (300, 400, 800, 1200, 1500, 1700, 2800):
        heat = min(slope * power + intercept for slope, intercept in lines)
        assert heat == pytest.approx(heat_at(points, power), rel=1e-9), power
    for power in (600, 1000, 1400, 1600, 2200):
        heat = min(slope * power + intercept for slope, intercept in lines)
        assert heat <= heat_at(points, power) + 1e-9, power


def test_every_watt_buys_less_heat_the_harder_it_runs():
    lines = heat_lines(parse_modulation(MEASURED), 300, 2800)
    slopes = [slope for slope, _ in lines]
    assert slopes == sorted(slopes, reverse=True)
    assert slopes[0] == pytest.approx(1.0) and slopes[-1] < 0.3


def test_a_dent_is_bridged_never_below_the_curve():
    points = [(500, 1.0), (1000, 0.6), (1500, 0.9)]  # 1000 W is a dent
    lines = heat_lines(points, 500, 1500)
    assert len(lines) == 1
    slope, intercept = lines[0]
    for power, factor in points:
        assert slope * power + intercept >= factor * power - 1e-9


def test_nothing_to_model_without_points_or_range():
    assert heat_lines([], 300, 2800) == []
    assert heat_lines(parse_modulation(MEASURED), 2800, 2800) == []
