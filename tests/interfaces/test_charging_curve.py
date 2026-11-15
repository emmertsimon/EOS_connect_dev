"""
The charging curve as the user writes it, and as the live limit and the plan read it.

The three consumers - config validation, the battery interface and the optimizer
backend - all go through these functions, so a disagreement between them shows up here
first.
"""

import pytest

from src.interfaces.charging_curve import (
    ChargingCurveError,
    builtin_curve_points,
    builtin_power_at,
    concave_envelope,
    concave_pieces,
    curve_power_at,
    envelope_deviation,
    envelope_exact_up_to,
    envelope_segments,
    full_range_points,
    limited_pieces,
    parse_curve_points,
    validate_curve_text,
)

MAX_W = 6000


# -- parsing -----------------------------------------------------------------------------


def test_points_in_watts_are_read_and_sorted():
    points = parse_curve_points("100:800, 0:6000, 95:3000, 90:6000", MAX_W)
    assert points == [(0.0, 6000.0), (90.0, 6000.0), (95.0, 3000.0), (100.0, 800.0)]


def test_power_can_be_a_share_of_the_maximum_or_in_kw():
    points = parse_curve_points("90:100%, 95:1.5kW, 100:10 %", MAX_W)
    assert points == [(90.0, 6000.0), (95.0, 1500.0), (100.0, 600.0)]


def test_a_decimal_comma_works_when_points_are_split_by_semicolons_or_lines():
    assert parse_curve_points("90:6000; 92,5:4000\n100:500", MAX_W) == [
        (90.0, 6000.0), (92.5, 4000.0), (100.0, 500.0),
    ]


def test_with_plain_commas_a_comma_always_separates_points():
    """'90:6000,5:3000' is two points, never 90 % then 6000.5 W."""
    assert parse_curve_points("90:6000,5:3000", MAX_W) == [(5.0, 3000.0), (90.0, 6000.0)]


def test_power_above_the_maximum_is_capped_to_it():
    assert parse_curve_points("50:9000", MAX_W) == [(50.0, 6000.0)]


def test_empty_text_means_no_points():
    assert parse_curve_points("", MAX_W) == []
    assert parse_curve_points("   ", MAX_W) == []
    assert parse_curve_points(None, MAX_W) == []


def test_pairs_are_accepted_as_a_list_too():
    assert parse_curve_points([[100, 800], [0, 6000]], MAX_W) == [(0.0, 6000.0), (100.0, 800.0)]


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("abc", "not a point"),
        ("90-6000", "not a point"),
        ("110:300", "between 0 and 100"),
        ("90:-5", "must not be negative"),
        ("90:6000, 90:5000", "given twice"),
        ("50:150%", "more than 100 %"),
    ],
)
def test_bad_points_are_refused_with_a_message_naming_them(text, fragment):
    with pytest.raises(ChargingCurveError, match=fragment):
        parse_curve_points(text, MAX_W)
    assert fragment in validate_curve_text(text, MAX_W)


def test_validation_passes_a_good_curve_even_without_a_known_maximum():
    assert validate_curve_text("0:100%, 90:100%, 100:15%") == ""


# -- interpolation -----------------------------------------------------------------------


CURVE = [(0.0, 6000.0), (90.0, 6000.0), (95.0, 3000.0), (100.0, 800.0)]


@pytest.mark.parametrize(
    "soc, power",
    [(0, 6000), (50, 6000), (90, 6000), (92.5, 4500), (95, 3000), (97.5, 1900), (100, 800)],
)
def test_power_is_interpolated_linearly(soc, power):
    assert curve_power_at(CURVE, soc) == pytest.approx(power)


def test_the_curve_is_held_flat_outside_its_points():
    points = [(20.0, 4000.0), (80.0, 2000.0)]
    assert curve_power_at(points, 5) == 4000.0
    assert curve_power_at(points, 95) == 2000.0


def test_a_single_point_is_a_constant_limit():
    assert curve_power_at([(70.0, 2500.0)], 10) == 2500.0
    assert curve_power_at([(70.0, 2500.0)], 99) == 2500.0


def test_the_builtin_formula_is_unchanged():
    """The numbers EOS Connect has always used: full to 50 %, quadratic taper, 0.05 C."""
    assert builtin_power_at(40, 3000, 10000) == pytest.approx(3000)
    assert builtin_power_at(80, 3000, 10000) == pytest.approx(0.3 * (1 - 30 / 60) ** 2 * 10000)
    assert builtin_power_at(100, 3000, 10000) == pytest.approx(500)


def test_the_builtin_samples_follow_the_formula():
    for soc, power in builtin_curve_points(6000, 16000):
        assert power == pytest.approx(builtin_power_at(soc, 6000, 16000))


# -- what the optimizer gets -------------------------------------------------------------


def test_points_are_padded_flat_to_both_ends():
    assert full_range_points([(20.0, 4000.0), (80.0, 2000.0)]) == [
        (0.0, 4000.0), (20.0, 4000.0), (80.0, 2000.0), (100.0, 2000.0),
    ]


def test_a_concave_curve_is_its_own_envelope():
    concave = [(0.0, 6000.0), (80.0, 6000.0), (90.0, 5000.0), (100.0, 1000.0)]
    assert concave_envelope(concave) == concave
    assert envelope_deviation(concave) == pytest.approx(0.0)


def test_a_convex_stretch_is_bridged_and_reported():
    # 90 -> 95 drops 3000 W, 95 -> 100 only 2200 W: the curve bends upward there.
    assert concave_envelope(full_range_points(CURVE)) == [
        (0.0, 6000.0), (90.0, 6000.0), (100.0, 800.0),
    ]
    # The hull passes 95 % at 3400 W instead of 3000 W.
    assert envelope_deviation(CURVE) == pytest.approx(400.0)


def test_every_segment_line_lies_on_or_above_the_envelope():
    """The minimum over the lines is the envelope, and no line can go below it."""
    points = [(10.0, 2000.0), (20.0, 6000.0), (90.0, 6000.0), (100.0, 1000.0)]
    segments = envelope_segments(points)
    hull = concave_envelope(full_range_points(points))
    for soc in range(0, 101):
        lines = [p0 + slope * (soc - s0) for s0, p0, slope in segments]
        assert min(lines) == pytest.approx(curve_power_at(hull, soc))
        assert min(lines) >= 0


def test_a_flat_curve_is_one_flat_segment():
    assert envelope_segments([(50.0, 3000.0)]) == [(0.0, 3000.0, 0.0)]


SIMON = "0:6000, 90:6000, 91:3200, 93:1600, 95:1000, 97:700, 98:500"


def test_a_concave_curve_is_one_piece_and_exact_everywhere():
    concave = [(0.0, 6000.0), (80.0, 6000.0), (90.0, 5000.0), (100.0, 1000.0)]
    assert len(concave_pieces(concave)) == 1
    assert envelope_exact_up_to(concave) == 100.0


def test_a_curve_is_cut_where_it_bends_upward():
    pieces = concave_pieces(parse_curve_points(SIMON, MAX_W))
    assert [(lo, hi) for lo, hi, _ in pieces] == [
        (0.0, 91.0), (91.0, 93.0), (93.0, 95.0), (95.0, 98.0), (98.0, 100.0),
    ]
    # Each piece is the minimum of its own lines, which reproduces the curve.
    points = parse_curve_points(SIMON, MAX_W)
    for soc in (50, 90.5, 92, 94, 96, 97.5, 99):
        lo, hi, lines = next(p for p in pieces if p[0] <= soc <= p[1])
        assert min(p0 + k * (soc - s0) for s0, p0, k in lines) == pytest.approx(
            curve_power_at(points, soc)
        )


def test_the_envelope_is_exact_up_to_the_first_upward_bend():
    assert envelope_exact_up_to(parse_curve_points(SIMON, MAX_W)) == 90.0


def test_merging_keeps_the_cuts_that_best_preserve_the_charging_time():
    points = parse_curve_points(SIMON, MAX_W)
    merged = limited_pieces(points, 3)
    assert [(lo, hi) for lo, hi, _ in merged] == [(0.0, 93.0), (93.0, 98.0), (98.0, 100.0)]
    # A merged piece follows the hull of its points: generous inside, never below.
    for soc in (91, 92, 94, 95, 97):
        lo, hi, lines = next(p for p in merged if p[0] <= soc <= p[1])
        assert min(p0 + k * (soc - s0) for s0, p0, k in lines) >= curve_power_at(points, soc) - 1e-6
    assert limited_pieces(points, 9) == concave_pieces(points)
