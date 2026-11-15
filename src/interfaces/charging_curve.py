"""
State-of-charge dependent charge power - the battery's charging curve.

A battery does not take its full charge power all the way to 100 %: the BMS tapers it
as the cells fill. EOS Connect knows that in two places, and both have to agree:

- the live limit written to the inverter for the slot that is running, and
- the plan, which decides how much the battery can still absorb in each slot ahead.

The curve is either the built-in formula (full power to 50 %, then a quadratic taper
to 0.05 C) or a list of points the user enters - state of charge against power,
interpolated linearly in between. Points are what a real battery needs: a pack that
holds full power to 90 % and then drops hard is not something a single formula can
describe, and neither is one that also tapers at the bottom.

Everything here is a pure function on plain numbers so the battery interface, the
config validation and the optimizer backend all read the same curve.
"""

import itertools
import re

# Shown in the UI and in every parse error, so the format is never a guess.
CURVE_FORMAT_HELP = (
    "comma-separated 'SOC:power' points, SOC in %, power in W or in % of the maximum "
    "charge power - e.g. '0:6000, 90:6000, 95:3000, 100:800' or "
    "'0:100%, 90:100%, 95:50%, 100:10%'"
)

# The built-in formula is sampled at these SOCs when the plan needs it as points.
# Below 50 % it is flat, so one point at 0 % carries that part.
_BUILTIN_SAMPLE_SOCS = (0.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0,
                        95.0, 100.0)
_BUILTIN_MIN_C_RATE = 0.05

_POINT_RE = re.compile(
    r"^\s*(?P<soc>[+-]?\d+(?:[.,]\d+)?)\s*%?\s*[:=]\s*"
    r"(?P<power>[+-]?\d+(?:[.,]\d+)?)\s*(?P<unit>%|w|kw)?\s*$",
    re.IGNORECASE,
)


class ChargingCurveError(ValueError):
    """The text does not describe a usable charging curve."""


def _number(text):
    return float(text.replace(",", "."))


def _split_points(text):
    """
    Split the text into its points.

    Points are separated by commas - or, for anyone who writes a decimal comma
    ('92,5:3000'), by semicolons or line breaks: as soon as one of those appears,
    commas are read as decimal marks. Deciding per text rather than per comma keeps
    '90:6000,5:3000' from silently meaning something else.
    """
    if ";" in text or "\n" in text:
        parts = re.split(r"[;\n]", text)
    else:
        parts = text.split(",")
    return [p for p in (s.strip() for s in parts) if p]


def parse_curve_points(text, max_charge_power_w):
    """
    Parse the user's points into ``[(soc_percent, power_w), ...]``, sorted by SOC.

    Power may be given in W (the default), kW, or in % of *max_charge_power_w*.
    Values above the maximum charge power are capped to it - the curve can only
    take power away, never add it. An empty text means "no points" and returns [].

    Raises:
        ChargingCurveError: with a message that names the offending point.
    """
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        raw = [f"{p[0]}:{p[1]}" for p in text]
    else:
        text = str(text).strip()
        if not text:
            return []
        raw = _split_points(text)

    max_w = float(max_charge_power_w or 0)
    points = {}
    for item in raw:
        match = _POINT_RE.match(item)
        if not match:
            raise ChargingCurveError(
                f"'{item}' is not a point - expected {CURVE_FORMAT_HELP}"
            )
        soc = _number(match.group("soc"))
        power = _number(match.group("power"))
        unit = (match.group("unit") or "w").lower()
        if not 0.0 <= soc <= 100.0:
            raise ChargingCurveError(f"'{item}': SOC must be between 0 and 100 %")
        if power < 0:
            raise ChargingCurveError(f"'{item}': power must not be negative")
        if unit == "%":
            if power > 100.0:
                raise ChargingCurveError(f"'{item}': more than 100 % of the maximum")
            if max_w <= 0:
                raise ChargingCurveError(
                    f"'{item}': a percentage needs the maximum charge power to be set"
                )
            power = max_w * power / 100.0
        elif unit == "kw":
            power *= 1000.0
        if soc in points:
            raise ChargingCurveError(f"SOC {soc:g} % is given twice")
        points[soc] = min(power, max_w) if max_w > 0 else power

    return sorted(points.items())


def validate_curve_text(text, max_charge_power_w=None):
    """Return '' when *text* parses, otherwise the error message (for the config API)."""
    try:
        # Without a known maximum a percentage cannot be resolved, so validate the
        # syntax against a nominal one; the real cap is applied at runtime.
        parse_curve_points(text, max_charge_power_w or 100000)
    except ChargingCurveError as exc:
        return str(exc)
    return ""


def curve_power_at(points, soc_percent):
    """
    Charge power at *soc_percent*, interpolated linearly between the points.

    Below the first point and above the last the curve is held flat at that point's
    value, so a single point is a constant limit.
    """
    if not points:
        raise ChargingCurveError("the curve has no points")
    soc = float(soc_percent)
    if soc <= points[0][0]:
        return float(points[0][1])
    for (s0, p0), (s1, p1) in zip(points, points[1:]):
        if soc <= s1:
            return float(p0 + (p1 - p0) * (soc - s0) / (s1 - s0))
    return float(points[-1][1])


def builtin_power_at(soc_percent, max_charge_power_w, capacity_wh):
    """
    The built-in curve: full power up to 50 %, then a quadratic taper to 0.05 C.

    This is the formula EOS Connect has always used; it lives here so the live limit
    and the plan cannot drift apart.
    """
    if capacity_wh <= 0:
        return float(max_charge_power_w)
    max_c_rate = float(max_charge_power_w) / float(capacity_wh)
    soc = float(soc_percent)
    if soc <= 50:
        c_rate = max_c_rate
    else:
        c_rate = max(_BUILTIN_MIN_C_RATE, max_c_rate * (1 - (soc - 50) / 60) ** 2)
    return min(c_rate * capacity_wh, float(max_charge_power_w))


def builtin_curve_points(max_charge_power_w, capacity_wh):
    """The built-in formula sampled as points, for consumers that need a curve."""
    return [
        (soc, builtin_power_at(soc, max_charge_power_w, capacity_wh))
        for soc in _BUILTIN_SAMPLE_SOCS
    ]


def concave_envelope(points):
    """
    The smallest concave curve on or above *points* (their upper hull).

    A linear program can hold "charge power <= curve(SOC)" exactly only when the
    curve is concave: then it is the minimum of the lines through its segments, and
    each line is one inequality. A curve with a convex stretch - a hard drop at 90 %
    followed by a long tail - would need integer variables per slot to be exact. The
    hull keeps the problem linear and errs on the generous side, so the plan can
    over-estimate a convex stretch slightly but never forbids a charge the battery
    could take. :func:`envelope_deviation` reports how far off that is.
    """
    pts = sorted((float(s), float(p)) for s, p in points)
    hull = []
    for p in pts:
        # Pop while the last two points and p make a non-right turn (upper hull).
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            cross = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def full_range_points(points):
    """
    *points* held flat out to 0 % and 100 %, the way :func:`curve_power_at` reads them.

    The optimizer works with the lines through the segments, which - unlike the
    interpolation - do not stop at the first and last point. Padding first makes both
    describe the same curve over the whole SOC range, and keeps a curve that rises
    from its first point from turning into a negative limit near empty.
    """
    pts = sorted((float(s), float(p)) for s, p in points)
    if not pts:
        raise ChargingCurveError("the curve has no points")
    if pts[0][0] > 0.0:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] < 100.0:
        pts.append((100.0, pts[-1][1]))
    return pts


def envelope_deviation(points):
    """Largest amount (W) by which the concave hull exceeds the points' curve."""
    pts = full_range_points(points)
    hull = concave_envelope(pts)
    worst = 0.0
    for soc, power in pts:
        worst = max(worst, curve_power_at(hull, soc) - float(power))
    return worst


def envelope_segments(points):
    """
    The hull as ``[(soc_from, power_from, slope_w_per_percent), ...]``.

    Every segment extended to a full line is an upper bound on the charge power; the
    minimum over all of them is the hull, which lies on or below every one of those
    lines across the whole SOC range - so none of them can demand a negative charge.
    A flat curve yields a single segment with slope 0.
    """
    hull = concave_envelope(full_range_points(points))
    # Collinear neighbours were already merged by the hull; a fully flat curve leaves
    # its two end points.
    if len(hull) == 1:
        return [(hull[0][0], hull[0][1], 0.0)]
    segments = []
    for (s0, p0), (s1, p1) in zip(hull, hull[1:]):
        segments.append((s0, p0, (p1 - p0) / (s1 - s0)))
    return segments


def concave_pieces(points, tolerance_w_per_percent=1e-6):
    """
    Split the curve into stretches that are each concave on their own.

    A concave stretch is the minimum of the lines through its segments, which a
    linear program holds exactly. Where the curve bends upward - the slope gets
    *less* steep - a new stretch begins. A concave curve comes back as one piece;
    a curve like '90:6000, 91:3200, 93:1600, 95:1000' as several, and the solver
    then decides per slot which stretch the state of charge is in.

    Returns ``[(soc_from, soc_to, [(soc_anchor, power, slope_w_per_percent), ...]),
    ...]`` over the full 0-100 % range (see :func:`full_range_points`).
    """
    pts = full_range_points(points)
    # Drop points that sit on a straight line between their neighbours.
    slopes = [(p1 - p0) / (s1 - s0) for (s0, p0), (s1, p1) in zip(pts, pts[1:])]
    pieces = []
    start = 0
    for i in range(1, len(slopes)):
        if slopes[i] > slopes[i - 1] + tolerance_w_per_percent:
            pieces.append((start, i))
            start = i
    pieces.append((start, len(slopes)))
    result = []
    for first, last in pieces:
        segment_lines = []
        for k in range(first, last):
            line = (pts[k][0], pts[k][1], slopes[k])
            if segment_lines and abs(segment_lines[-1][2] - line[2]) <= tolerance_w_per_percent:
                continue  # collinear with the previous segment
            segment_lines.append(line)
        result.append((pts[first][0], pts[last][0], segment_lines))
    return result


def envelope_exact_up_to(points, tolerance_w=1e-6):
    """
    The SOC up to which the concave envelope *is* the curve.

    Below it a plan built on the envelope is exact; above it the envelope can promise
    more than the battery takes. A concave curve returns 100.
    """
    pts = full_range_points(points)
    hull = concave_envelope(pts)
    exact_to = pts[0][0]
    for soc, power in pts:
        if curve_power_at(hull, soc) - power > tolerance_w:
            return exact_to
        if any(abs(soc - h_soc) <= 1e-9 for h_soc, _ in hull):
            exact_to = soc
    return 100.0


def _hull_lines(points):
    """Segment lines of the concave hull of *points* (which span one piece)."""
    hull = concave_envelope(points)
    if len(hull) == 1:
        return [(hull[0][0], hull[0][1], 0.0)]
    return [(s0, p0, (p1 - p0) / (s1 - s0)) for (s0, p0), (s1, p1) in zip(hull, hull[1:])]


def _piece_value(pieces, soc):
    """Power of a list of pieces at *soc*: the minimum of the lines of its piece."""
    for _, hi, lines in pieces:
        if soc <= hi + 1e-9:
            return min(p + k * (soc - s0) for s0, p, k in lines)
    lines = pieces[-1][2]
    return min(p + k * (soc - s0) for s0, p, k in lines)


def limited_pieces(points, max_pieces):
    """
    :func:`concave_pieces`, merged down to at most *max_pieces*.

    Every extra piece costs the solver a binary per slot, and the solve time grows
    quickly with it. When the curve has more upward bends than that, neighbouring
    pieces are merged and each merged piece follows the concave hull of its points -
    generous inside, like the envelope, but only there. The cuts kept are the ones
    that best preserve how long the battery takes to fill: the error is measured as
    charging time over the stretches the merge touches.
    """
    exact = concave_pieces(points)
    if max_pieces < 1 or len(exact) <= max_pieces:
        return exact
    pts = full_range_points(points)
    cuts = [lo for lo, _, _ in exact[1:]]
    # Merging can only make the curve more generous where it bends upward, which
    # starts where the envelope stops being exact.
    start = envelope_exact_up_to(points)
    grid = [start + (100.0 - start) * (j + 0.5) / 400 for j in range(400)]
    truth = [max(curve_power_at(pts, soc), 1e-6) for soc in grid]

    def build(kept):
        bounds = [0.0] + list(kept) + [100.0]
        pieces = []
        for lo, hi in zip(bounds, bounds[1:]):
            inside = [(soc, w) for soc, w in pts if lo - 1e-9 <= soc <= hi + 1e-9]
            pieces.append((lo, hi, _hull_lines(inside)))
        return pieces

    def lost_time(pieces):
        return sum(
            1.0 / w - 1.0 / max(_piece_value(pieces, soc), 1e-6)
            for soc, w in zip(grid, truth)
        )

    candidates = [build(kept) for kept in itertools.combinations(cuts, max_pieces - 1)]
    return min(candidates, key=lost_time)
