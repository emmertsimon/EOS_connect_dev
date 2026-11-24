"""
Learning a house's heat capacity, heat loss and free heat from how well it forecast.

The calibrator learns a store from windows - each a span over which the temperature
moved a measurable amount, turned into one equation of the energy balance with the
capacity taken as known. That works for a pool. It does not work for a house with floor
heating, and for a reason the windows cannot see: the heat a run puts in goes into the
screed first and reaches the rooms over the next hours, so a window during a run sees
less warming than the heat meter says, and the window after it sees warming with the
heat pump off. On a real 170 m2 house the windows explained 8 % of what they were fitted
to, the loss was held at its configured guess, and the capacity was never learned at
all - it is taken from a table of construction types.

Both were wrong in the direction that hurts. With 21 kWh/K and 51 W/K the house cooled,
in the model, at half the rate it did in the rooms; every plan expected the morning to
hold above the comfort floor until noon, the rooms reached it at nine, and the plan
started the heat pump there and then at the morning price.

What the planner needs is not an energy balance per window but a forecast that holds
for the next hours. So that is what is fitted: from every hour of the recorded history,
the model is run forward twelve hours on the heat that was actually delivered, the air
it actually stood in and the sun that actually shone, and the four numbers that make
those forecasts closest to what the rooms then did are the house's:

    C · dT/dt  =  Q_heat  +  g · S  +  G  -  H · (T - T_air)

- ``C`` the heat stored per kelvin, Wh/K - the capacity;
- ``H`` the heat lost per kelvin to the air, W/K - loss coefficient times surface;
- ``g`` the heat gained per watt of the irradiance proxy (the PV meter, else forecast);
- ``G`` the heat the house makes for itself, W - people, appliances, a hot water tank.

Run over twelve hours the screed's delay averages out: a model that put the heat
straight into the rooms with the true capacity would overshoot during a run and
undershoot after it, and the forecast error is what tells it so. Two coupled stores
(rooms and screed) were tried on the same data and forecast no better.

A learned set replaces the configured one only if it forecast the last two days - which
it did not see while being fitted - better than the set in use. Otherwise the house
keeps what it had. Each number is pulled weakly towards its configured value so that a
direction the data cannot tell apart (more loss and more free heat, say) stays where
the owner put it instead of wandering.

Pure numpy, no state: the calibrator owns the samples and the result.
"""

import math

import numpy as np

# The grid the history is put on, the forecast length and how often one starts.
STEP_SECONDS = 900
HORIZON_SLOTS = 48
START_EVERY_SLOTS = 4

# The tail held back to judge a fit by, and how much history a fit needs before it.
HOLDOUT_HOURS = 48.0
MIN_TRAIN_HOURS = 72.0
# At least this many forecasts must start in the held-back tail.
MIN_HOLDOUT_STARTS = 24
# The heating must have delivered something worth calling a signal.
MIN_HEAT_KWH = 10.0

# A gap in the samples longer than this leaves the slot it falls in unknown.
MAX_SAMPLE_GAP_SECONDS = 20 * 60
# A slot counts when its samples cover at least this share of it.
MIN_SLOT_COVERAGE = 0.6

# A learned set has to forecast the held-back tail this much better than the one in
# use. Two per cent is below what a week of weather changes on its own, so a set is
# not swapped for one that is merely different.
ACCEPT_RATIO = 0.98

# Physical bounds, as factors of the configured value for capacity and loss and as
# absolutes for the gains. Outside them the fit has found a different house, not this one.
CAPACITY_FACTOR_BOUNDS = (0.33, 4.0)
LOSS_FACTOR_BOUNDS = (0.33, 4.0)
SOLAR_GAIN_BOUNDS = (0.0, 5.0)
INTERNAL_GAIN_BOUNDS_W = (0.0, 1500.0)

# How far each number may move from the configured one before the pull towards it
# matters, and how strongly it pulls: one spread costs as much as (PRIOR_K)^2 of mean
# squared forecast error - a few per cent of a typical fit, so the data decides
# wherever it can.
CAPACITY_SPREAD_LN = math.log(2.0)
LOSS_SPREAD_LN = math.log(2.0)
SOLAR_GAIN_SPREAD = 0.5
INTERNAL_GAIN_SPREAD_W = 300.0
PRIOR_K = 0.02

# Levenberg-Marquardt: iterations and the step at which it counts as settled.
MAX_ITERATIONS = 60
TOLERANCE = 1e-7


def _seconds(moment):
    return moment.timestamp()


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def grid(samples, cop=None):
    """
    The recorded samples on a fifteen-minute grid.

    Returns ``(start, temperature, air, heat_w, solar_w)`` as numpy arrays, NaN where a
    slot is unknown, and the start of the first slot - or None without enough samples.

    The temperature and the air are the slot's mean reading. The heat and the sun are
    what accumulated over it: the heat meter and the PV meter differenced sample to
    sample, so a counter that updates in coarse steps is still exact over the slot. A
    sample pair without a usable heat reading falls back to the electrical power times
    *cop* at that air, where *cop* is given; without it the slot is unknown. A pair
    further apart than `MAX_SAMPLE_GAP_SECONDS` says nothing about what happened between.
    """
    rows = []
    for sample in samples or ():
        if not isinstance(sample, dict) or sample.get("timestamp") is None:
            continue
        medium = _number(sample.get("medium_c"))
        air = _number(sample.get("ambient_c"))
        if medium is None or air is None:
            continue
        rows.append((
            _seconds(sample["timestamp"]), medium, air,
            _number(sample.get("heat_counter_kwh")),
            _number(sample.get("pv_counter_kwh")),
            _number(sample.get("solar_w")) or 0.0,
            _number(sample.get("power_w")) or 0.0,
            _number(sample.get("cop_air_c")),
        ))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda row: row[0])

    first = math.floor(rows[0][0] / STEP_SECONDS) * STEP_SECONDS
    count = int((rows[-1][0] - first) // STEP_SECONDS) + 1
    temp_sum = np.zeros(count)
    temp_n = np.zeros(count)
    air_sum = np.zeros(count)
    heat_wh = np.zeros(count)
    solar_wh = np.zeros(count)
    covered = np.zeros(count)

    for row in rows:
        index = int((row[0] - first) // STEP_SECONDS)
        temp_sum[index] += row[1]
        air_sum[index] += row[2]
        temp_n[index] += 1

    for previous, current in zip(rows, rows[1:]):
        gap = current[0] - previous[0]
        if gap <= 0 or gap > MAX_SAMPLE_GAP_SECONDS:
            continue
        hours = gap / 3600.0
        heat = None
        if previous[3] is not None and current[3] is not None:
            delta = current[3] - previous[3]
            if delta >= 0:
                heat = delta * 1000.0
        if heat is None and cop is not None:
            air = previous[7] if previous[7] is not None else previous[2]
            try:
                heat = max(0.0, previous[6]) * hours * float(cop(air))
            except (TypeError, ValueError):
                heat = None
        if heat is None:
            continue
        sun = None
        if previous[4] is not None and current[4] is not None:
            delta = current[4] - previous[4]
            if delta >= 0:
                sun = delta * 1000.0
        if sun is None:
            sun = max(0.0, previous[5]) * hours
        index = int(((previous[0] + current[0]) / 2.0 - first) // STEP_SECONDS)
        if 0 <= index < count:
            heat_wh[index] += heat
            solar_wh[index] += sun
            covered[index] += gap

    with np.errstate(invalid="ignore", divide="ignore"):
        temperature = np.where(temp_n > 0, temp_sum / temp_n, np.nan)
        air = np.where(temp_n > 0, air_sum / temp_n, np.nan)
        known = covered >= MIN_SLOT_COVERAGE * STEP_SECONDS
        scale = np.where(known, STEP_SECONDS / np.maximum(covered, 1.0), np.nan)
        hours = STEP_SECONDS / 3600.0
        heat_w = heat_wh * scale / hours
        solar_w = solar_wh * scale / hours
    return first, temperature, air, heat_w, solar_w


def _starts(temperature, lo, hi):
    """Forecast start slots in ``[lo, hi)`` that have a reading to start from."""
    return np.array(
        [s for s in range(lo, hi, START_EVERY_SLOTS) if math.isfinite(temperature[s])],
        dtype=int,
    )


def forecast_errors(params, data, starts):
    """
    Forecast minus measurement, ``(len(starts), HORIZON_SLOTS)``, NaN where unknown.

    A forecast stops at the first slot whose inputs are unknown; its later steps are
    left out rather than guessed.
    """
    capacity, loss, solar_gain, internal_w = params
    _, temperature, air, heat_w, solar_w = data
    count = len(temperature)
    hours = STEP_SECONDS / 3600.0
    state = temperature[starts].astype(float)
    alive = np.isfinite(state)
    errors = np.full((len(starts), HORIZON_SLOTS), np.nan)
    for step in range(HORIZON_SLOTS):
        index = starts + step
        inside = index + 1 < count
        index = np.where(inside, index, count - 2)
        drive = heat_w[index] + solar_gain * solar_w[index] + internal_w
        drive = drive - loss * (state - air[index])
        alive &= inside & np.isfinite(drive)
        state = np.where(alive, state + hours * drive / capacity, state)
        measured = temperature[index + 1]
        errors[:, step] = np.where(alive & np.isfinite(measured), state - measured, np.nan)
    return errors


def _score(errors):
    """RMS over every known forecast step, mean at the last one, and how many."""
    known = errors[np.isfinite(errors)]
    if known.size == 0:
        return None, None, 0
    last = errors[:, -1]
    last = last[np.isfinite(last)]
    return (
        float(np.sqrt(np.mean(known ** 2))),
        float(np.mean(last)) if last.size else None,
        int(known.size),
    )


class _Space:
    """The four numbers in the space they are fitted in: logs for the positive scales."""

    def __init__(self, prior):
        self.prior = prior
        c0, h0 = prior["capacity_wh_per_k"], prior["loss_w_per_k"]
        self.low = np.array([
            math.log(c0 * CAPACITY_FACTOR_BOUNDS[0]),
            math.log(h0 * LOSS_FACTOR_BOUNDS[0]),
            SOLAR_GAIN_BOUNDS[0], INTERNAL_GAIN_BOUNDS_W[0],
        ])
        self.high = np.array([
            math.log(c0 * CAPACITY_FACTOR_BOUNDS[1]),
            math.log(h0 * LOSS_FACTOR_BOUNDS[1]),
            SOLAR_GAIN_BOUNDS[1], INTERNAL_GAIN_BOUNDS_W[1],
        ])
        self.centre = self.clip(self.encode(prior))
        self.spread = np.array([
            CAPACITY_SPREAD_LN, LOSS_SPREAD_LN, SOLAR_GAIN_SPREAD, INTERNAL_GAIN_SPREAD_W,
        ])
        # Finite-difference steps, a small share of each spread.
        self.step = self.spread * 1e-3

    @staticmethod
    def encode(values):
        return np.array([
            math.log(values["capacity_wh_per_k"]),
            math.log(values["loss_w_per_k"]),
            float(values.get("solar_gain", 0.0) or 0.0),
            float(values.get("internal_gain_w", 0.0) or 0.0),
        ])

    @staticmethod
    def decode(x):
        return (math.exp(x[0]), math.exp(x[1]), float(x[2]), float(x[3]))

    def clip(self, x):
        return np.minimum(np.maximum(x, self.low), self.high)


def _residuals(x, space, data, starts):
    """Forecast errors scaled to an RMS, then the pull towards the configured values."""
    errors = forecast_errors(space.decode(x), data, starts)
    known = errors[np.isfinite(errors)]
    if known.size == 0:
        return None
    pull = PRIOR_K * (x - space.centre) / space.spread
    return np.concatenate([known / math.sqrt(known.size), pull])


def _fit(space, data, starts, x0):
    """Bounded Levenberg-Marquardt on `_residuals`. Returns the best point found."""
    x = space.clip(np.array(x0, dtype=float))
    r = _residuals(x, space, data, starts)
    if r is None:
        return None
    cost = float(r @ r)
    damping = 1e-3
    for _ in range(MAX_ITERATIONS):
        jacobian = np.empty((len(r), len(x)))
        for column in range(len(x)):
            shifted = x.copy()
            shifted[column] += space.step[column]
            moved = _residuals(shifted, space, data, starts)
            if moved is None or len(moved) != len(r):
                return x
            jacobian[:, column] = (moved - r) / space.step[column]
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ r
        improved = False
        for _ in range(12):
            system = normal + damping * np.diag(np.diag(normal) + 1e-12)
            try:
                delta = -np.linalg.solve(system, gradient)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            candidate = space.clip(x + delta)
            trial = _residuals(candidate, space, data, starts)
            if trial is not None and len(trial) == len(r):
                trial_cost = float(trial @ trial)
                if trial_cost < cost:
                    settled = cost - trial_cost < TOLERANCE * max(cost, 1e-12)
                    x, r, cost = candidate, trial, trial_cost
                    damping = max(damping / 3.0, 1e-9)
                    improved = True
                    break
            damping *= 4.0
        if not improved or settled:
            break
    return x


def fit_house(samples, prior, current=None, cop=None):
    """
    Fit capacity, loss and the two free gains to the recorded history.

    *prior* holds the configured values (``capacity_wh_per_k``, ``loss_w_per_k``,
    ``solar_gain``, ``internal_gain_w``) - the centre of the pull and the anchor of the
    bounds. *current* holds the set in use, the one a new fit has to beat; the prior
    when omitted. *cop* is the fallback for slots without a heat reading.

    Returns a result dict whether or not the new set was good enough: ``accepted`` says
    which, ``reason`` why, and the four numbers are the new set when accepted and the
    current one otherwise. None when the history cannot support a fit at all.
    """
    current = dict(current or prior)
    data = grid(samples, cop=cop)
    if data is None:
        return None
    first, temperature, air, heat_w, _ = data
    count = len(temperature)
    hours = STEP_SECONDS / 3600.0
    known_hours = float(np.sum(np.isfinite(heat_w) & np.isfinite(temperature))) * hours
    heat_kwh = float(np.nansum(heat_w)) * hours / 1000.0

    holdout_slots = int(HOLDOUT_HOURS * 3600 / STEP_SECONDS)
    split = count - holdout_slots
    result = {
        "fitted_at": None,
        "hours_used": round(known_hours, 1),
        "heat_kwh": round(heat_kwh, 1),
        "accepted": False,
    }
    if split - HORIZON_SLOTS <= 0 or known_hours < MIN_TRAIN_HOURS + HOLDOUT_HOURS:
        return {**result, "reason": "not enough history yet"}
    if heat_kwh < MIN_HEAT_KWH:
        return {**result, "reason": "too little heat delivered to learn from"}

    train = _starts(temperature, 0, split - HORIZON_SLOTS)
    holdout = _starts(temperature, split, count - 1)
    if len(train) < MIN_HOLDOUT_STARTS or len(holdout) < MIN_HOLDOUT_STARTS:
        return {**result, "reason": "not enough forecasts to judge a fit by"}

    space = _Space(prior)
    start = space.clip(space.encode(current))
    trained = _fit(space, data, train, start)
    if trained is None:
        return {**result, "reason": "the history has no usable forecasts"}

    in_use = space.decode(space.clip(space.encode(current)))
    rms_new, bias_new, _ = _score(forecast_errors(space.decode(trained), data, holdout))
    rms_old, bias_old, _ = _score(forecast_errors(in_use, data, holdout))
    result.update({
        "holdout_rms_k": None if rms_new is None else round(rms_new, 3),
        "holdout_bias_12h_k": None if bias_new is None else round(bias_new, 3),
        "holdout_rms_in_use_k": None if rms_old is None else round(rms_old, 3),
        "holdout_bias_12h_in_use_k": None if bias_old is None else round(bias_old, 3),
    })
    if rms_new is None or rms_old is None:
        return {**result, "reason": "the held-back days have no usable forecasts"}
    if rms_new >= ACCEPT_RATIO * rms_old:
        return {**_named(in_use), **result,
                "reason": "did not forecast the last two days better than the values in use"}

    # Good enough: settle it on the whole history, the held-back days included.
    final = _fit(space, data, _starts(temperature, 0, count - 1), trained)
    if final is None:
        final = trained
    rms_all, bias_all, _ = _score(
        forecast_errors(space.decode(final), data, _starts(temperature, 0, count - 1))
    )
    result.update({
        "accepted": True,
        "reason": "forecast the last two days better than the values in use",
        "rms_k": None if rms_all is None else round(rms_all, 3),
        "bias_12h_k": None if bias_all is None else round(bias_all, 3),
    })
    return {**_named(space.decode(final)), **result}


def _named(params):
    capacity, loss, solar_gain, internal_w = params
    return {
        "capacity_wh_per_k": round(capacity, 1),
        "loss_w_per_k": round(loss, 2),
        "solar_gain": round(solar_gain, 4),
        "internal_gain_w": round(internal_w, 1),
    }


def plausible(values, prior):
    """Whether a restored set lies inside the bounds the fit itself would allow."""
    try:
        capacity = float(values["capacity_wh_per_k"])
        loss = float(values["loss_w_per_k"])
        solar_gain = float(values.get("solar_gain", 0.0) or 0.0)
        internal_w = float(values.get("internal_gain_w", 0.0) or 0.0)
    except (KeyError, TypeError, ValueError):
        return False
    c0, h0 = prior["capacity_wh_per_k"], prior["loss_w_per_k"]
    return (
        c0 * CAPACITY_FACTOR_BOUNDS[0] <= capacity <= c0 * CAPACITY_FACTOR_BOUNDS[1]
        and h0 * LOSS_FACTOR_BOUNDS[0] <= loss <= h0 * LOSS_FACTOR_BOUNDS[1]
        and SOLAR_GAIN_BOUNDS[0] <= solar_gain <= SOLAR_GAIN_BOUNDS[1]
        and INTERNAL_GAIN_BOUNDS_W[0] <= internal_w <= INTERNAL_GAIN_BOUNDS_W[1]
    )
