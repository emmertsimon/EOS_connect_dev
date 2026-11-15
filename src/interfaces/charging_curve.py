"""
State-of-charge dependent charge power - the battery's charging curve.

A battery does not take its full charge power all the way to 100 %: the BMS tapers it
as the cells fill. EOS Connect knows that in two places, and both have to agree:

- the live limit written to the inverter for the slot that is running, and
- the plan, which decides how much the battery can still absorb in each slot ahead.

Both read it through ``BatteryInterface.charge_power_limit_w``. The curve is either
the built-in formula there (full power to 50 %, then a quadratic taper to 0.05 C) or
a list of points the user enters - state of charge against power, interpolated
linearly in between. Points are what a real battery needs: a pack that holds full
power to 90 % and then drops hard is not something a single formula can describe,
and neither is one that also tapers at the bottom.

Everything here is a pure function on plain numbers so the battery interface and the
config validation read the points the same way.
"""

import re

# Shown in the UI and in every parse error, so the format is never a guess.
CURVE_FORMAT_HELP = (
    "comma-separated 'SOC:power' points, SOC in %, power in W or in % of the maximum "
    "charge power - e.g. '0:6000, 90:6000, 95:3000, 100:800' or "
    "'0:100%, 90:100%, 95:50%, 100:10%'"
)

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


def parse_curve_points(text, max_charge_power_w):  # pylint: disable=too-many-branches
    """
    Parse the user's points into ``[(soc_percent, power_w), ...]``, sorted by SOC.

    Power may be given in W (the default), kW, or in % of *max_charge_power_w*.
    Values above the maximum charge power are capped to it - the curve can only
    take power away, never add it. An empty text means "no points" and returns [];
    an empty point (a trailing separator, '0:6000,') is skipped. Each SOC may be
    given once: a vertical step at one SOC is refused, two points half a percent
    apart describe it.

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
            # One power per SOC: a vertical step has no slope the plan could work
            # with. A step a fraction of a percent wide does the same job.
            low, high = (soc, soc + 0.5) if soc <= 99.5 else (soc - 0.5, soc)
            raise ChargingCurveError(
                f"SOC {soc:g} % is given twice - for a step, give the two powers half "
                f"a percent apart, e.g. '{low:g}:{points[soc]:.0f}, {high:g}:{power:.0f}'"
            )
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
