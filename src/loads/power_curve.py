"""
The power a modulating heat pump draws while it runs, as a formula learned from data.

A modulating heat pump is not something a plan can set to a power. Its own controller
picks the compressor speed from the weather-compensation curve and from how far the
water and the rooms still have to go, and a release signal only decides *whether* it
runs. Measured on a Bosch CS6800iAW over 35 heating runs (EMS-ESP, Nov 2026): at
-1 C outside and a house at 21.0 C it ran flat out at ~2.2 kW; three hours later, at
-3 C but with the house at 21.4 C, it had settled to ~650 W by itself; at 8-12 C it
never drew more than ~650 W. Planning such a pump at a power of the optimizer's
choosing - the house heated at 300 W on paper while the meter read 2.2 kW - got the
battery, the grid and the heat wrong at once.

So the power is a property of the slot, not a decision: the plan only switches the
pump on or off, and while it is on it draws::

    P = base + air * T_air + store * (T_store - ref)

held between the appliance's minimum and its rated power. ``T_air`` is the air the
COP is worked out from (outdoors for an air-source pump), ``T_store`` the store's own
temperature at the start of the slot (the house), ``ref`` a fixed reference so that
``base`` reads as "the draw at 0 C outside with the store at ref". For the house above
the fit came to ``base:1810, air:-109, store:-880, ref:21`` - 109 W more per kelvin
colder outside, 880 W less per kelvin warmer inside.

The configured coefficients are the starting point; the samples the manager records
anyway (power, air, store temperature every cycle) refine them. The configured curve
counts as `PRIOR_SAMPLES` measurements spread across a wide range of air and store
temperatures, so a fresh install follows the configuration, a week of runs at 8-12 C
moves the curve where it was measured and leaves the cold end near the configuration
until a frost has been seen. A shut-off learner (``power_curve_learning: false``) uses
the configuration as it is.

Pure numbers and numpy; the model reads the config, the solver reads per-slot powers.
"""

import math
import re
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

FORMAT_HELP = (
    "comma-separated 'name:value' terms - base (W at 0 C air with the store at ref), "
    "air (W per K of air), store (W per K of the store above ref), ref (C) - e.g. "
    "'base:1810, air:-109, store:-880, ref:21'"
)

# Names accepted for each coefficient. "temp" and "house" read more naturally for a
# house; "outdoor" for an air-source pump.
_ALIASES = {
    "base": "base", "p0": "base",
    "air": "air", "outdoor": "air", "outside": "air", "ambient": "air",
    "store": "store", "temp": "store", "house": "store", "indoor": "store",
    "ref": "ref", "reference": "ref",
}

_TERM_RE = re.compile(
    r"^\s*(?P<name>[a-z_0-9]+)\s*[:=]\s*(?P<value>[-+]?\d+(?:[.,]\d+)?)\s*$",
    re.IGNORECASE,
)

# The configured curve weighs as this many measurements, spread over the grid below -
# air across a heating season, the store half a kelvin either side of ref. Spread that
# wide it holds a slope only as firmly as the history's own spread does not: a week
# across 16 K of weather outweighs it, a week of mild days at 8-12 C leaves the frost
# end close to the configuration.
PRIOR_SAMPLES = 12
PRIOR_AIR_C = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
PRIOR_STORE_K = (-0.5, 0.0, 0.5)

# A compressor ramps up after a start and its first minutes say nothing about the
# draw it settles at: samples this soon after the run began are not learned from.
RUN_SETTLE_SECONDS = 10 * 60
# A gap this long between two running samples means the start was not seen.
RUN_GAP_SECONDS = 15 * 60
# Samples kept for the fit - two weeks of five-minute samples while running, about the
# retention of the sample store the history is replayed from.
MAX_SAMPLES = 4000
# Residuals beyond this count with falling weight (Huber): a defrost, a hot-water
# change-over or a reading taken across a switch is one bad minute, not a new law.
HUBER_W = 300.0
FIT_ITERATIONS = 12
# A reading this far above the rating is a bad sensor, not the pump.
IMPLAUSIBLE_FACTOR = 2.0
# How long the planned store temperature takes to settle from where it is now back to
# target, for the first guess of the per-slot power - the solver then refines it from
# the temperatures it actually plans (see Optimizer._refine_power_curves).
GUESS_RELAX_HOURS = 6.0


class PowerCurveError(ValueError):
    """The text does not describe a usable power curve."""


def _number(text):
    return float(str(text).replace(",", "."))


def parse_power_curve(text):
    """
    Parse ``name:value`` terms into ``{"base": .., "air": .., "store": .., "ref": ..}``.

    Only the names given are returned; an empty text returns {}. Terms are separated by
    commas, or - for a decimal comma - by semicolons or line breaks.

    Raises:
        PowerCurveError: with a message that names the offending term.
    """
    if text is None:
        return {}
    if isinstance(text, dict):
        raw = [f"{key}:{value}" for key, value in text.items()]
    else:
        text = str(text).strip()
        if not text:
            return {}
        parts = re.split(r"[;\n]", text) if (";" in text or "\n" in text) else text.split(",")
        raw = [part.strip() for part in parts if part.strip()]
    terms = {}
    for item in raw:
        match = _TERM_RE.match(item)
        if not match:
            raise PowerCurveError(f"'{item}' is not a term - expected {FORMAT_HELP}")
        name = _ALIASES.get(match.group("name").lower())
        if name is None:
            raise PowerCurveError(f"'{match.group('name')}' is not a term - expected {FORMAT_HELP}")
        if name in terms:
            raise PowerCurveError(f"'{name}' is given twice")
        terms[name] = _number(match.group("value"))
    if "ref" in terms and not -50.0 <= terms["ref"] <= 150.0:
        raise PowerCurveError("ref must be a temperature between -50 and 150 C")
    return terms


def validate_power_curve_text(text):
    """Return '' when *text* parses, otherwise the error message (for the config API)."""
    try:
        parse_power_curve(text)
    except PowerCurveError as exc:
        return str(exc)
    return ""


@dataclass(frozen=True)
class PowerCurve:
    """
    ``P = base + air * T_air + store * (T_store - ref)``, held in ``[min_w, max_w]``.

    Attributes:
        base_w: Draw at 0 C air with the store at ``ref_c``.
        air_w_per_k: Change per kelvin of air temperature (negative for heating: colder
            air, more power).
        store_w_per_k: Change per kelvin of store temperature above ``ref_c`` (negative
            for a house: warmer rooms, the controller throttles).
        ref_c: Reference store temperature.
        min_w: The least the appliance draws while running.
        max_w: The most - its rated power.
    """

    base_w: float
    air_w_per_k: float = 0.0
    store_w_per_k: float = 0.0
    ref_c: float = 20.0
    min_w: float = 0.0
    max_w: float = 0.0

    def raw(self, air_c, store_c):
        """The formula without the bounds."""
        return (
            self.base_w
            + self.air_w_per_k * float(air_c)
            + self.store_w_per_k * (float(store_c) - self.ref_c)
        )

    def clamp(self, power_w):
        """*power_w* held between the bounds."""
        high = self.max_w if self.max_w > 0 else float("inf")
        return min(high, max(self.min_w, float(power_w)))

    def at(self, air_c, store_c):
        """The draw at this air and store temperature, within the bounds."""
        return self.clamp(self.raw(air_c, store_c))

    def coefficients(self):
        """``(base_w, air_w_per_k, store_w_per_k)`` - what the learner fits."""
        return (self.base_w, self.air_w_per_k, self.store_w_per_k)

    def with_coefficients(self, base_w, air_w_per_k, store_w_per_k):
        """The same curve - bounds and reference - with other coefficients."""
        return replace(
            self, base_w=float(base_w), air_w_per_k=float(air_w_per_k),
            store_w_per_k=float(store_w_per_k),
        )

    def as_text(self):
        """The curve in the format of the ``power_curve`` setting."""
        return (
            f"base:{self.base_w:.0f}, air:{self.air_w_per_k:.1f}, "
            f"store:{self.store_w_per_k:.0f}, ref:{self.ref_c:g}"
        )

    def as_dict(self):
        """The curve for the card and the API."""
        return {
            "base_w": round(self.base_w, 1),
            "air_w_per_k": round(self.air_w_per_k, 2),
            "store_w_per_k": round(self.store_w_per_k, 1),
            "ref_c": round(self.ref_c, 2),
            "min_w": round(self.min_w, 1),
            "max_w": round(self.max_w, 1),
            "text": self.as_text(),
        }


def curve_from_config(terms, rated_power_w, min_power_w, default_ref_c):
    """
    The configured curve: the terms given, the rest neutral.

    Without a ``base`` the appliance starts out at its rated power everywhere - the
    behaviour from before there were curves - and the learner moves it from there.
    """
    rated = max(0.0, float(rated_power_w or 0.0))
    floor = max(0.0, float(min_power_w or 0.0))
    if floor <= 0 or floor >= rated:
        # Running at nothing is not running: without a minimum, a tenth of the rating.
        floor = 0.1 * rated
    return PowerCurve(
        base_w=float(terms.get("base", rated)),
        air_w_per_k=float(terms.get("air", 0.0)),
        store_w_per_k=float(terms.get("store", 0.0)),
        ref_c=float(terms.get("ref", default_ref_c)),
        min_w=floor,
        max_w=rated,
    )


class PowerCurveLearner:
    """
    Refines a configured power curve from the samples the manager records.

    Each sample carries the metered power, the air temperature and the store
    temperature. Running samples, once the compressor has settled, enter a robust,
    bounds-aware least-squares fit; the configured curve enters it as `PRIOR_SAMPLES`
    measurements spread over a wide grid, which keeps the fit where data is missing.
    Readings at the rated power while the curve already asks for more than that, or at
    the minimum while it asks for less, say only "at the limit" and are left out.

    The fit is rebuilt from the recorded history at startup (``observe_series``), so
    nothing of its own needs persisting.
    """

    def __init__(self, prior, idle_power_w=50.0, learn=True):
        self.prior = prior
        self.idle_power_w = max(1.0, float(idle_power_w or 0.0) or 50.0)
        self.learn = bool(learn)
        self.samples = deque(maxlen=MAX_SAMPLES)
        self._run_started_at = None
        self._last_running_at = None
        self._fitted = None
        self._stats = {}
        self._dirty = True

    # -- configuration ------------------------------------------------------------------

    def set_prior(self, prior, idle_power_w=None, learn=None):
        """New configured curve and bounds; the samples are kept and refitted."""
        if prior != self.prior:
            self.prior = prior
            self._dirty = True
        if idle_power_w is not None:
            self.idle_power_w = max(1.0, float(idle_power_w or 0.0) or 50.0)
        if learn is not None and bool(learn) != self.learn:
            self.learn = bool(learn)
            self._dirty = True

    # -- observation --------------------------------------------------------------------

    def observe(self, sample):
        """
        Take one recorded sample; True when it entered the fit.

        A sample needs ``timestamp``, ``power_w``, ``medium_c`` and an air temperature
        (``cop_air_c``, else ``ambient_c``). Not running, still settling after a start,
        or implausible: noted for the run detection and otherwise skipped.
        """
        if not isinstance(sample, dict):
            return False
        moment = sample.get("timestamp")
        power = _finite(sample.get("power_w"))
        store = _finite(sample.get("medium_c"))
        air = _finite(sample.get("cop_air_c"))
        if air is None:
            air = _finite(sample.get("ambient_c"))
        if moment is None or power is None:
            return False

        if power < self.idle_power_w:
            self._run_started_at = None
            self._last_running_at = None
            return False
        if (
            self._last_running_at is None
            or _seconds(self._last_running_at, moment) > RUN_GAP_SECONDS
            or _seconds(self._last_running_at, moment) < 0
        ):
            # The first running sample seen: the start was at most a cycle ago.
            self._run_started_at = moment
        self._last_running_at = moment
        if _seconds(self._run_started_at, moment) < RUN_SETTLE_SECONDS:
            return False
        if store is None or air is None:
            return False
        if self.prior.max_w > 0 and power > IMPLAUSIBLE_FACTOR * self.prior.max_w:
            return False
        self.samples.append((air, store, power))
        self._dirty = True
        return True

    def observe_series(self, samples):
        """Replay a recorded history, oldest first; returns how many were learned."""
        rows = [s for s in samples or [] if isinstance(s, dict) and s.get("timestamp")]
        rows.sort(key=lambda s: s["timestamp"])
        return sum(1 for sample in rows if self.observe(sample))

    # -- the curve ------------------------------------------------------------------------

    def curve(self):
        """The curve in force: the fit when learning, otherwise the configuration."""
        if not self.learn:
            return self.prior
        if self._dirty:
            self._refit()
        return self._fitted or self.prior

    def source(self):
        """Where the curve in force comes from: "configured" or "learned"."""
        if not self.learn or not self.samples:
            return "configured"
        return "learned"

    def _refit(self):
        self._dirty = False
        self._stats = {}
        if not self.samples:
            self._fitted = None
            return
        prior = self.prior
        data = np.asarray(self.samples, dtype=float)
        x_data = np.column_stack(
            [np.ones(len(data)), data[:, 0], data[:, 1] - prior.ref_c]
        )
        y_data = data[:, 2]

        grid = [(air, prior.ref_c + dk) for air in PRIOR_AIR_C for dk in PRIOR_STORE_K]
        x_prior = np.array([[1.0, air, store - prior.ref_c] for air, store in grid])
        y_prior = np.array([prior.raw(air, store) for air, store in grid])
        w_prior = np.full(len(grid), PRIOR_SAMPLES / len(grid))

        theta = np.array(prior.coefficients(), dtype=float)
        low, high = prior.min_w, (prior.max_w if prior.max_w > 0 else np.inf)
        used = np.ones(len(data), dtype=bool)
        for _ in range(FIT_ITERATIONS):
            linear = x_data @ theta
            # A reading at a limit the curve already passes says only "at the limit".
            used = ~(
                ((y_data >= 0.97 * high) & (linear >= high))
                | ((y_data <= 1.03 * low) & (linear <= low))
            )
            residual = np.abs(y_data - linear)
            weight = np.where(residual <= HUBER_W, 1.0, HUBER_W / np.maximum(residual, 1e-9))
            weight = weight * used
            weighted = x_data * weight[:, None]
            weighted_prior = x_prior * w_prior[:, None]
            normal = weighted.T @ x_data + weighted_prior.T @ x_prior
            rhs = weighted.T @ y_data + weighted_prior.T @ y_prior
            try:
                updated = np.linalg.solve(normal, rhs)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(updated)):
                break
            converged = np.max(np.abs(updated - theta)) < 0.5
            theta = updated
            if converged:
                break

        # Plausible magnitudes only: a fit that wants more than a fifth of the rating per
        # kelvin of air, or twice the rating per kelvin of store, is not physics.
        scale = prior.max_w if prior.max_w > 0 else max(abs(prior.base_w), 1000.0)
        theta[1] = float(np.clip(theta[1], -scale / 5.0, scale / 5.0))
        theta[2] = float(np.clip(theta[2], -2.0 * scale, 2.0 * scale))
        fitted = prior.with_coefficients(*theta)

        predicted = np.array([fitted.at(air, store) for air, store, _ in data])
        error = y_data - predicted
        spread = float(np.var(y_data))
        self._fitted = fitted
        self._stats = {
            "rmse_w": round(float(np.sqrt(np.mean(error ** 2))), 1),
            "mae_w": round(float(np.mean(np.abs(error))), 1),
            "r2": round(1.0 - float(np.var(error)) / spread, 3) if spread > 0 else None,
            "air_span_k": round(_span(data[:, 0]), 1),
            "store_span_k": round(_span(data[:, 1]), 2),
            "at_limit": int(len(data) - int(np.sum(used))),
        }

    def state(self):
        """What the card and the API show."""
        curve = self.curve()
        return {
            "source": self.source(),
            "learning": self.learn,
            "samples": len(self.samples),
            "curve": curve.as_dict(),
            "configured": self.prior.as_dict(),
            **({"fit": dict(self._stats)} if self._stats and self.learn else {}),
        }


def guess_store_c(current_c, target_c, hours_ahead):
    """
    Where the store will roughly be *hours_ahead* from now: back towards target.

    Only the first guess of the per-slot power; a solver that follows the temperature
    replaces it with the temperatures it plans.
    """
    if target_c is None:
        return float(current_c)
    return float(target_c) + (float(current_c) - float(target_c)) * math.exp(
        -max(0.0, float(hours_ahead)) / GUESS_RELAX_HOURS
    )


def factor_table(points, min_w, max_w, steps=24):
    """
    The modulation COP factor (see `loads.modulation`) tabulated across the curve's
    range, as ``[[power_w, factor], ...]`` - linear between rows, which is what the
    solver interpolates when it moves a slot's power. [] without points.
    """
    from .modulation import factor_at  # pylint: disable=import-outside-toplevel

    if not points or max_w <= 0:
        return []
    low = max(1.0, float(min_w))
    high = max(low, float(max_w))
    if high == low:
        return [[round(low, 1), round(factor_at(points, low), 4)]]
    step = (high - low) / steps
    return [
        [round(low + i * step, 1), round(factor_at(points, low + i * step), 4)]
        for i in range(steps + 1)
    ]


def _span(values):
    """The range between the 5th and the 95th percentile."""
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _seconds(earlier, later):
    try:
        return (later - earlier).total_seconds()
    except (TypeError, AttributeError):
        return 0.0
