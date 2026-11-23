"""
A modulating appliance's COP across its power range, as the user writes it and as the
planner reads it.
"""

import pytest

from src.loads.modulation import (
    ModulationError,
    factor_at,
    heat_at,
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
