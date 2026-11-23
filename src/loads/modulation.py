"""
How a modulating appliance's efficiency changes with the power it draws.

A modulating heat pump does not run at one power, and it does not deliver the same
heat per watt across its range: at low modulation the compressor works against a small
temperature lift and the COP is at its best, at full power the flow runs hotter and
every extra watt buys less heat. Which power it runs at is its own controller's choice
(see `loads.power_curve`); this module says what that power does to the COP.

The user describes it as ``W:factor`` points - electrical power against a factor on
the COP the load's model already works out per slot (from the air temperature): e.g.
``400:1.0, 800:1.0, 1200:0.93, 1500:0.81, 1700:0.74``. The planner scales each slot's
COP by the factor at the power the curve gives that slot, and the calibrator divides
a heat-meter reading by the factor at the power it was taken at.

Pure functions on plain numbers.
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
