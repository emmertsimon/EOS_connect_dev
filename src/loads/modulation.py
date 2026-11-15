"""
How a modulating appliance's efficiency changes with the power it draws.

A modulating heat pump does not run at one power. It draws anything between its
minimum and its maximum, and it does not deliver the same heat per watt across that
range: at low modulation the compressor works against a small temperature lift and
the COP is at its best, at full power the flow runs hotter and every extra watt buys
less heat. A plan that thinks one watt is as good as the next puts the whole day's
heat into the cheapest hour at full power; the real optimum is often a long run at
low power.

The user describes it as ``W:factor`` points - electrical power against a factor on
the COP the load's model already works out per slot (from the air temperature): e.g.
``400:1.0, 800:1.0, 1200:0.93, 1500:0.81, 1700:0.74``. This module turns them into
what the solver needs: the heat a slot gets for its electricity, as the minimum of a
few lines, which keeps it linear.

Pure functions on plain numbers; the model reads the config, the solver reads lines.
"""

import re

FORMAT_HELP = (
    "comma-separated 'W:factor' points, power in W, factor on the COP - e.g. "
    "'400:1.0, 800:1.0, 1200:0.93, 1500:0.81, 1700:0.74'"
)

_POINT_RE = re.compile(
    r"^\s*(?P<power>\d+(?:[.,]\d+)?)\s*(?P<unit>w|kw)?\s*[:=]\s*"
    r"(?P<factor>\d+(?:[.,]\d+)?)\s*$",
    re.IGNORECASE,
)


class ModulationError(ValueError):
    """The text does not describe a usable modulation curve."""


def _number(text):
    return float(text.replace(",", "."))


def parse_modulation(text):
    """
    Parse ``W:factor`` points into ``[(power_w, factor), ...]``, sorted by power.

    Points are separated by commas, or - for a decimal comma - by semicolons or line
    breaks. Power may be given in kW. An empty text means "no curve" and returns [].

    Raises:
        ModulationError: with a message that names the offending point.
    """
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        raw = [f"{p[0]}:{p[1]}" for p in text]
    else:
        text = str(text).strip()
        if not text:
            return []
        parts = re.split(r"[;\n]", text) if (";" in text or "\n" in text) else text.split(",")
        raw = [p.strip() for p in parts if p.strip()]
    points = {}
    for item in raw:
        match = _POINT_RE.match(item)
        if not match:
            raise ModulationError(f"'{item}' is not a point - expected {FORMAT_HELP}")
        power = _number(match.group("power"))
        if (match.group("unit") or "w").lower() == "kw":
            power *= 1000.0
        factor = _number(match.group("factor"))
        if power <= 0:
            raise ModulationError(f"'{item}': power must be above 0 W")
        if not 0.05 <= factor <= 3.0:
            raise ModulationError(f"'{item}': factor must be between 0.05 and 3")
        if power in points:
            raise ModulationError(f"{power:g} W is given twice")
        points[power] = factor
    return sorted(points.items())


def validate_modulation_text(text):
    """Return '' when *text* parses, otherwise the error message (for the config API)."""
    try:
        parse_modulation(text)
    except ModulationError as exc:
        return str(exc)
    return ""


def heat_at(points, power_w):
    """
    Heat at *power_w*, in units of the per-slot COP: ``factor * power`` on the points.

    Between points the factor is interpolated linearly; below the first it is held.
    Above the last point the heat keeps growing at the rate of the last stretch rather
    than at the last factor: the efficiency keeps falling where nobody measured it,
    instead of jumping back up to the average of the range below.
    """
    if not points:
        return float(power_w)
    if power_w <= points[0][0]:
        return float(points[0][1]) * power_w
    for (p0, f0), (p1, f1) in zip(points, points[1:]):
        if power_w <= p1:
            return (f0 + (f1 - f0) * (power_w - p0) / (p1 - p0)) * power_w
    if len(points) == 1:
        return float(points[0][1]) * power_w
    (p0, f0), (p1, f1) = points[-2], points[-1]
    rate = (f1 * p1 - f0 * p0) / (p1 - p0)
    return f1 * p1 + max(0.0, rate) * (power_w - p1)


def factor_at(points, power_w):
    """The COP factor at *power_w* (see :func:`heat_at`)."""
    if power_w <= 0:
        return float(points[0][1]) if points else 1.0
    return heat_at(points, power_w) / power_w


def heat_lines(points, min_power_w, max_power_w):
    """
    The heat a slot gets per Wh drawn, as lines for the solver.

    Heat (in units of the per-slot COP) against electrical power is ``factor * P``.
    The lines join the curve's points between *min_power_w* and *max_power_w* along
    their smallest concave cover, each as ``(slope, intercept_w)``: the solver holds the
    heat under ``slope * E + intercept_w * hours * on`` for every line. For a curve that
    is concave already - efficiency falling with power, the usual case - the lines meet
    it at every point and run a little under it in between; a dent is bridged, so the
    plan may count on a little more heat there.

    Returns [] when there is nothing to model (no points, or a single power).
    """
    if not points or max_power_w <= 0:
        return []
    low = max(0.0, float(min_power_w))
    high = float(max_power_w)
    if high <= low:
        return []
    powers = sorted({low, high, *[p for p, _ in points if low < p < high]})
    if low <= 0:
        powers = [p for p in powers if p > 0] or [high]
    heat = [(p, heat_at(points, p)) for p in powers]
    if len(heat) == 1:
        p, h = heat[0]
        return [(h / p, 0.0)]
    # Upper concave hull (monotone chain), left to right.
    hull = []
    for point in heat:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(point)
    lines = []
    for (x0, y0), (x1, y1) in zip(hull, hull[1:]):
        slope = (y1 - y0) / (x1 - x0)
        lines.append((slope, y0 - slope * x0))
    return lines
