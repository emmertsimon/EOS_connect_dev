"""
The demand model behind every stored-heat appliance.

A pool heat pump, a sauna, a hot water tank and a buffer tank pose the same question -
how much electrical energy does this store need over the next two days, and when is it
allowed to take it - and differ only in the numbers. So there is one model, and the
appliance types are presets over it.

Two answers come out, and keeping them apart is what makes the feature behave:

- ``total_wh`` is the **forecast**: the energy the store will need across the whole
  horizon, including the losses it will suffer while sitting at target. It stays honest
  even when the store is momentarily satisfied, because the optimizer is planning two
  days ahead and a pool that is at temperature right now will still need topping up
  tonight.
- ``detail["active"]`` is the **control** flag: whether the appliance should be running
  at all at this moment, with a deadband so it is not switched on for the last tenth of
  a degree.

Confusing the two produces either a forecast that collapses to zero whenever the pump
is off, or a pump that hunts around its setpoint.
"""

import logging
import math
from datetime import timedelta

from .. import efficiency as slot_efficiency
from .base import KIND_CONTINGENT, BaseDemandModel, EnergyDemand
from .calibration import IDLE_POWER_W, ThermalCalibrator
from .cover_habit import OBSERVED_HOURS, CoverHabit
from ..modulation import ModulationError, factor_at, parse_modulation
from ..power_curve import (
    PowerCurveError,
    PowerCurveLearner,
    curve_from_config,
    factor_table,
    guess_store_c,
    parse_power_curve,
)
from .thermal_physics import (
    WH_PER_M3_PER_K,
    cop_at,
    energy_to_raise_wh,
    loss_power_w,
    thermal_to_electrical_wh,
)

logger = logging.getLogger("__main__")

# Reasons reported alongside the demand, shown on MQTT and the dashboard.
REASON_NO_TEMPERATURE = "temperature sensor unavailable"
REASON_AT_TARGET = "at target temperature"
REASON_IN_DEADBAND = "within deadband - not starting"
REASON_FROST = "frost protection"
REASON_SOFT_LIMIT = "below soft limit - charging in the next allowed window"
REASON_HEATING = "below target temperature"
REASON_HOLDING = "covering standing losses"
REASON_OUT_OF_SEASON = "out of season"
REASON_CIRCUIT_OFF = "heating circuit off"

# What a circuit sensor reads while the heat pump's own controller has switched the
# circuit off - the summer shutdown of a weather-compensated controller, for instance.
DEFAULT_CIRCUIT_OFF_STATES = ("off",)
# Readings that say nothing about the circuit. A bus gateway that drops out for a minute
# every twenty must not flip the plan, so the last definite reading stands this long.
CIRCUIT_UNKNOWN_STATES = ("", "unavailable", "unknown", "none")
CIRCUIT_HOLD_SECONDS = 30 * 60
# The same gateway drops the store's thermometer too. For this long the last reading
# stands in, so a one-minute gap does not throw the plan away for a cycle.
TEMPERATURE_HOLD_SECONDS = 15 * 60
# A tank sitting on its soft limit reads 41.9 and 42.0 by turns. Once under a limit,
# the store counts as under it until it is this far above, so the plan does not come
# and go with the last digit of the sensor.
DEFAULT_LIMIT_HYSTERESIS_K = 0.3

# How a store is planned. By energy, the original way: an amount over the horizon,
# placed by price. By temperature: the store itself, handed to a solver that can follow
# it slot by slot - see `store`.
PLAN_BY_ENERGY = "energy"
PLAN_BY_TEMPERATURE = "temperature"
PLANNING_MODES = (PLAN_BY_ENERGY, PLAN_BY_TEMPERATURE)

# How far above target a store planned by temperature may be heated to bank cheap or
# free energy, unless configured.
DEFAULT_STORAGE_HEADROOM_K = 1.0

# Per-slot exclusions, reported so a shortfall can name the setting behind it.
REASON_SLOT_WINDOW = "outside allowed hours"
REASON_SLOT_SEASON = "out of season"
REASON_SLOT_AMBIENT = "too cold to run"


def _as_float(value, default=None):
    """Sensor readings arrive as strings with units; take the number or give up."""
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    try:
        return float(str(value).strip().split()[0])
    except (ValueError, IndexError, AttributeError):
        return default


def _as_bool(value, default=True):
    """A config flag that may arrive as a string."""
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off")
    return bool(value)


def _round_or_none(value, digits=1):
    """Round a value that may not be there."""
    return None if value is None else round(value, digits)


def _parse_month_day(value):
    """``"04-15"`` to ``(4, 15)``, or None when unset or unparseable."""
    if not value:
        return None
    try:
        month, day = str(value).strip().split("-")[:2]
        return (int(month), int(day))
    except (ValueError, TypeError):
        logger.warning("[LOADS] cannot read %r as a MM-DD date - ignoring it", value)
        return None


def _in_season(day, start, end):
    """Whether a (month, day) falls in a season that may wrap the new year."""
    if start is None or end is None:
        return True
    if start <= end:
        return start <= day <= end
    return day >= start or day <= end


def _in_window(hour, start, end):
    """Whether an hour falls in an allowed window that may wrap midnight."""
    if start is None or end is None or start == end:
        return True
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def _parse_states(value, default):
    """A comma-separated list (or a list) of states, lower-cased; *default* if empty."""
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value or "").split(",")
    states = tuple(str(item).strip().lower() for item in items if str(item).strip())
    return states or tuple(default)


class ThermalStorageModel(BaseDemandModel):
    """Stored-heat demand with an online-calibrated loss coefficient and COP."""

    kind = KIND_CONTINGENT
    sensor_keys = ("temp_sensor", "target_temp_sensor", "power_sensor",
                   "ambient_temp_sensor", "cover_sensor", "heat_sensor",
                   "cop_air_sensor", "circuit_sensor")

    def __init__(self, entry_id, config, calibration=None):
        super().__init__(entry_id, config)
        self._apply_config(self.config)
        # What the calibrator may learn for this kind of store - see the preset. Kept so
        # a reset starts again under the same rules rather than the pool's.
        self.calibration_options = dict(calibration or {})
        self.calibrator = self._new_calibrator()
        self.power_learner = self._new_power_learner()
        self.cover_habit = CoverHabit()
        # The last definite reading of the circuit sensor, and when it was taken.
        self._circuit_off = None
        self._circuit_state = None
        self._circuit_seen_at = None
        # The last readable store temperature, and when it was read - see
        # `current_temperature`.
        self._temp_last = None
        self._temp_seen_at = None
        # Whether the store went under its soft or hard limit and has not yet come
        # back above it by the hysteresis - see `_limits`.
        self._soft_latched = False
        self._hard_latched = False

    def _new_calibrator(self):
        calibrator = ThermalCalibrator(
            volume_m3=self.volume_m3,
            surface_m2=self.surface_m2,
            loss_coefficient=self.config.get("heat_loss_w_per_m2_k", 25.0),
            cop_nominal=self.config.get("cop_nominal", 4.5),
            air_coefficient=self.config.get("cop_air_coeff", 0.0),
            cover_loss_factor=self.cover_loss_factor,
            idle_power_w=self.standby_threshold_w,
            **self.calibration_options,
        )
        calibrator.cop_factor = self._modulation_factor()
        return calibrator

    def _new_power_learner(self):
        return PowerCurveLearner(
            self.configured_power_curve,
            idle_power_w=self.standby_threshold_w or IDLE_POWER_W,
            learn=self.power_curve_learning,
        )

    def _modulation_factor(self):
        """
        The COP factor by electrical power, for the calibrator - or None.

        Only where the planner applies the curve too (the power follows a power curve):
        the level the calibrator learns is the COP at factor 1.0, and the planner
        scales it by the factor at each slot's power.
        """
        if not self.modulation or not self.power_curve_active:
            return None
        points = list(self.modulation)
        return lambda power_w: factor_at(points, power_w)

    def power_curve(self):
        """The power curve in force, or None for an appliance at its rated power."""
        if not self.power_curve_active:
            return None
        return self.power_learner.curve()

    def _apply_config(self, config):
        """
        Read the tunables out of a config dict.

        Shared by construction and hot reload. The physical dimensions are read here too
        because construction needs them, but they are restart-required in the schema:
        changing the volume of a store invalidates a calibration anchored to it, and
        quietly carrying the old coefficients forward would be worse than a restart.
        """
        self.volume_m3 = float(config.get("volume_m3", 0) or 0)
        self.surface_m2 = float(config.get("surface_m2", 0) or 0)
        self.rated_power_w = float(config.get("rated_power_w", 0) or 0)
        # A modulating heat pump: rated_power_w is then the most it draws, this the
        # least, and `modulation` how its COP changes in between (loads.modulation).
        self.min_power_w = max(0.0, float(config.get("min_power_w", 0) or 0))
        if self.min_power_w >= self.rated_power_w:
            self.min_power_w = 0.0
        try:
            self.modulation = parse_modulation(config.get("modulation_cop", ""))
        except ModulationError as exc:
            logger.warning(
                "[LOADS] '%s': modulation_cop ignored (%s) | Config: #managed-loads | "
                "ACTION REQUIRED", config.get("id", "?"), exc,
            )
            self.modulation = []
        # What such a pump draws is its controller's choice, not the plan's: the plan
        # switches it on and off, and while on it draws what the power curve says for
        # that slot's air and store temperature (loads.power_curve). Active for a
        # modulating appliance (a minimum power) or wherever a curve is configured.
        try:
            terms = parse_power_curve(config.get("power_curve", ""))
        except PowerCurveError as exc:
            logger.warning(
                "[LOADS] '%s': power_curve ignored (%s) | Config: #managed-loads | "
                "ACTION REQUIRED", config.get("id", "?"), exc,
            )
            terms = {}
        self.power_curve_active = bool(terms) or self.min_power_w > 0
        self.power_curve_learning = _as_bool(config.get("power_curve_learning", True))
        self.configured_power_curve = curve_from_config(
            terms, self.rated_power_w, self.min_power_w,
            default_ref_c=_as_float(config.get("target_temp"), 20.0),
        )
        self.deadband_k = max(0.0, float(config.get("deadband_k", 0.5) or 0.0))
        self.cover_loss_factor = float(config.get("cover_loss_factor", 1.0) or 1.0)
        # The permanent draw of the appliance's own systems - circulation pumps,
        # controls, an oil heating circuit - under which the meter never falls. It is
        # the on/off threshold, not a contribution: how the measured idling enters the
        # household forecast is decided where the plan is published.
        self.standby_threshold_w = max(
            0.0, float(config.get("standby_threshold_w", 0) or 0)
        )

        self.min_ambient_temp_c = config.get("min_ambient_temp_c")
        self.frost_protection_temp_c = config.get("frost_protection_temp_c")
        # The soft limit: below it the store must be charged within its next allowed
        # window whatever the price limit says - see `_must_run`. 0 or empty is off.
        soft = _as_float(config.get("soft_limit_temp_c"))
        self.soft_limit_temp_c = soft if soft is not None and soft > 0 else None
        self.soft_limit_hours = max(
            0.0, _as_float(config.get("soft_limit_hours"), 0.0) or 0.0
        )
        hysteresis = _as_float(config.get("limit_hysteresis_k"))
        self.limit_hysteresis_k = max(
            0.0, DEFAULT_LIMIT_HYSTERESIS_K if hysteresis is None else hysteresis
        )
        self.window_start = config.get("window_start")
        self.window_end = config.get("window_end")
        self.deadline_hours = config.get("deadline_hours")

        self.season_start = _parse_month_day(config.get("season_start"))
        self.season_end = _parse_month_day(config.get("season_end"))

        mode = str(config.get("planning_mode") or PLAN_BY_ENERGY).strip().lower()
        self.planning_mode = mode if mode in PLANNING_MODES else PLAN_BY_ENERGY
        try:
            headroom = float(config.get("storage_headroom_k", DEFAULT_STORAGE_HEADROOM_K))
        except (TypeError, ValueError):
            headroom = DEFAULT_STORAGE_HEADROOM_K
        self.storage_headroom_k = max(0.0, headroom)

        self.circuit_sensor = str(config.get("circuit_sensor") or "").strip()
        self.circuit_off_states = _parse_states(
            config.get("circuit_off_states"), DEFAULT_CIRCUIT_OFF_STATES
        )

    def reconfigure(self, config):
        """Apply changed settings, keeping the calibrator and everything it learned."""
        super().reconfigure(config)
        self._apply_config(self.config)
        # The threshold is a pure classification change: no window, row or coefficient
        # is invalidated by it, so the fit carries over.
        self.calibrator.set_idle_threshold(self.standby_threshold_w)
        self.calibrator.cop_factor = self._modulation_factor()
        # A new configured curve is a new prior; the samples behind the fit stay.
        self.power_learner.set_prior(
            self.configured_power_curve,
            idle_power_w=self.standby_threshold_w or IDLE_POWER_W,
            learn=self.power_curve_learning,
        )

    def reset_calibration(self):
        """Discard what has been learned and start again from the configured values."""
        self.calibrator = self._new_calibrator()
        self.power_learner = self._new_power_learner()
        self.cover_habit.reset()

    # -- configuration helpers ----------------------------------------------------------

    def target_temperature(self, readings):
        """The setpoint, from its sensor when one is configured, else from config."""
        from_sensor = _as_float(readings.get("target_temp_sensor"))
        if from_sensor is not None:
            return from_sensor
        return _as_float(self.config.get("target_temp"), 0.0)

    def is_covered(self, readings):
        """Whether the cover reads closed; anything unreadable means "open"."""
        raw = readings.get("cover_sensor")
        if raw is None:
            return False
        text = str(raw).strip().lower()
        return text in ("on", "true", "closed", "1", "yes", "home")

    def effective_cover_factor(self):
        """
        How much a closed cover actually cuts the losses.

        The configured value is the cold start; what the calibrator has since measured
        replaces it. A cover is the one part of this model the user has no way to look
        up - the number on a supplier's data sheet is for still air over new material -
        so guessing it once and holding it forever put its whole error into the loss
        coefficient instead.
        """
        if self.cover_loss_factor >= 1.0:
            return 1.0
        return self.calibrator.cover_loss_factor

    def standby_power_now(self, readings):
        """
        The permanent draw this meter is reading while the appliance is off, in watts.

        The threshold decides where standby ends; what is under it is measured, not
        configured, because the real draw - the pump's own figure is 80-120 W - moves
        about and the household forecast should carry what the meter actually says.
        None when there is no threshold, no readable power, or the appliance is above
        the threshold: there the standby is inside the running draw and is the
        schedule's business, not a separate household term.
        """
        if self.standby_threshold_w <= 0:
            return None
        power = _as_float(readings.get("power_sensor"), 0.0)
        if power is None or power <= 0 or power >= self.standby_threshold_w:
            return None
        return power

    def cover_factor(self, readings):
        """A closed cover cuts the losses; anything unreadable means "open"."""
        return self.effective_cover_factor() if self.is_covered(readings) else 1.0

    def is_running(self, readings):
        """
        Whether the appliance itself is drawing, above its permanent systems.

        The configured threshold says where the standby draw on this meter ends and the
        appliance begins; the 50 W default is the old guess for a circuit that does
        drop to nearly nothing. A heat pump idling at 100 W of circulation pumps and
        controls read as "running" around the clock under the bare guess, which held
        the release gate open inside the deadband and fed the calibration compressor
        windows that were only standby.
        """
        power = _as_float(readings.get("power_sensor"), 0.0)
        if power is None:
            return False
        # A configured threshold is authoritative, however low - the user knows the
        # circuit. Only the unset zero stands down to the bare noise-floor guess.
        return power >= (self.standby_threshold_w or IDLE_POWER_W)

    def circuit_off(self, ctx):
        """
        Whether the heat pump's own controller has switched this circuit off.

        A weather-compensated controller shuts its heating circuit down in summer mode,
        whatever the room thermostat or a release signal asks for. Planning heat into
        those hours only moves the battery around a load that cannot appear, and the
        release turns into a promise the pump does not keep. So a configured
        ``circuit_sensor`` reading one of ``circuit_off_states`` takes the whole horizon
        out until it reads otherwise; the next cycle after it comes back plans again.

        An unreadable sensor is not an off circuit: the last definite reading stands
        for half an hour, then the circuit counts as on, the state it had before the
        sensor existed. Returns ``(off, state_text)``.
        """
        if not self.circuit_sensor:
            return False, None
        raw = ctx.readings.get("circuit_sensor")
        text = "" if raw is None else str(raw).strip().lower()
        now = getattr(ctx, "now", None)
        if text in CIRCUIT_UNKNOWN_STATES:
            fresh = (
                self._circuit_seen_at is not None and now is not None
                and (now - self._circuit_seen_at).total_seconds() <= CIRCUIT_HOLD_SECONDS
            )
            if fresh and self._circuit_off is not None:
                return self._circuit_off, self._circuit_state
            return False, None
        off = text in self.circuit_off_states
        if off != bool(self._circuit_off):
            if off:
                logger.warning(
                    "[LOADS] '%s': heating circuit reads '%s' (%s) - the heat pump's "
                    "controller has switched it off, so nothing is planned or released "
                    "until it is back. A summer shutdown threshold set too low is the "
                    "usual cause | Config: #managed-loads",
                    self.id, text, self.circuit_sensor,
                )
            else:
                logger.info(
                    "[LOADS] '%s': heating circuit is back ('%s') - planning again",
                    self.id, text,
                )
        self._circuit_off = off
        self._circuit_state = text
        self._circuit_seen_at = now
        return off, text

    def current_temperature(self, ctx):
        """
        The store's temperature now, and how old it is.

        A bus gateway that drops out for a minute every twenty takes the thermometer
        with it, and a cycle that reads "unavailable" used to report no demand and
        throw away a soft-limit plan until the next one. So the last readable value
        stands in for `TEMPERATURE_HOLD_SECONDS`; after that the store counts as
        unreadable, as before. Returns ``(temperature_or_None, held_seconds_or_None)``
        - the age only when the value is a stand-in.
        """
        current = _as_float(ctx.readings.get("temp_sensor"))
        now = getattr(ctx, "now", None)
        if current is not None:
            self._temp_last = current
            self._temp_seen_at = now
            return current, None
        if self._temp_last is None or self._temp_seen_at is None or now is None:
            return None, None
        age = (now - self._temp_seen_at).total_seconds()
        if 0 <= age <= TEMPERATURE_HOLD_SECONDS:
            logger.debug(
                "[LOADS] '%s': temperature unreadable - using the last reading "
                "%.1f C from %.0f s ago", self.id, self._temp_last, age,
            )
            return self._temp_last, age
        return None, None

    def _limits(self, current):
        """
        Whether the store is under its hard and its soft limit, with hysteresis.

        Going under is immediate: at or below the hard limit, below the soft limit.
        Coming back is not: the store stays under a limit until it reads
        ``limit_hysteresis_k`` above it, so a tank hovering on 42.0 keeps its
        planned charge instead of losing it every other cycle. Returns
        ``(under_hard, under_soft)``.
        """
        band = self.limit_hysteresis_k
        hard = _as_float(self.frost_protection_temp_c)
        if hard is None:
            self._hard_latched = False
        elif current <= hard:
            self._hard_latched = True
        elif current > hard + band:
            self._hard_latched = False
        soft = self.soft_limit_temp_c
        if soft is None:
            self._soft_latched = False
        elif current < soft:
            self._soft_latched = True
        elif current >= soft + band:
            self._soft_latched = False
        return self._hard_latched, self._soft_latched

    # -- feasibility --------------------------------------------------------------------

    def feasibility(self, ctx):
        """
        Per-slot mask of when the appliance may run at all, and why not when it may not.

        Each slot's wall clock comes from the instant it starts, not from its index.
        Dividing the index gives the right answer on 364 days a year and is an hour
        out on the other two: a 25-hour day has 25 hourly slots, so from the
        transition onwards index // slots_per_hour names the hour before the real
        one — and a run window of 22:00–06:00 would open and close an hour late. One
        astimezone() per slot (192 at most) settles it exactly.
        """
        mask, reasons = [], []
        for index in range(ctx.slot_count):
            # astimezone() re-normalizes the pytz offset across a transition.
            slot_start = (
                ctx.anchor + timedelta(seconds=index * ctx.time_frame_base)
            ).astimezone(ctx.anchor.tzinfo)
            hour = slot_start.hour
            day = slot_start.date()

            reason = None
            if not _in_window(hour, self.window_start, self.window_end):
                reason = REASON_SLOT_WINDOW
            elif not _in_season((day.month, day.day), self.season_start, self.season_end):
                reason = REASON_SLOT_SEASON
            elif self.min_ambient_temp_c is not None:
                # The limit is about the air the appliance works in: for a heat pump
                # with its own outdoor air that is the air, not the tank's room.
                ambient = self._cop_air_at(ctx, index)
                if ambient is not None and ambient < float(self.min_ambient_temp_c):
                    reason = REASON_SLOT_AMBIENT

            mask.append(reason is None)
            reasons.append(reason)
        return mask, reasons

    @staticmethod
    def _ambient_at(ctx, index):
        if not ctx.ambient_temp_c:
            return None
        if index < len(ctx.ambient_temp_c):
            return ctx.ambient_temp_c[index]
        return ctx.ambient_temp_c[-1]

    @classmethod
    def _cop_air_at(cls, ctx, index):
        """
        The air the COP is worked out from: the heat pump's own, where one is
        configured (``cop_air_sensor``), otherwise the store's surroundings.
        """
        series = getattr(ctx, "cop_air_temp_c", None)
        if not series:
            return cls._ambient_at(ctx, index)
        if index < len(series):
            return series[index]
        return series[-1]

    # -- demand ---------------------------------------------------------------------------

    def demand(self, ctx):
        """Energy needed over the horizon, and whether to run right now."""
        current, held_s = self.current_temperature(ctx)
        if current is None:
            logger.debug(
                "[LOADS] '%s' has no readable temperature - reporting no demand", self.id
            )
            return EnergyDemand(
                kind=self.kind, total_wh=0.0, feasible=[False] * ctx.slot_count,
                max_power_w=self.rated_power_w, reason=REASON_NO_TEMPERATURE,
            )

        target = self.target_temperature(ctx.readings)
        cover_now = self.cover_factor(ctx.readings)
        cover = self.cover_series(ctx, cover_now)
        feasible, feasible_reason = self.feasibility(ctx)

        urgent, below_soft = self._limits(current)

        deadline_slot = None
        if self.deadline_hours:
            deadline_slot = min(
                ctx.slot_count - 1,
                ctx.current_slot
                + int(round(float(self.deadline_hours) * 3600 / ctx.time_frame_base)),
            )

        heat_up_wh = energy_to_raise_wh(self.volume_m3, target - current)
        losses_wh, mean_cop = self._horizon_losses(
            ctx, target, cover, feasible, until=deadline_slot
        )

        thermal_wh = max(0.0, heat_up_wh + losses_wh)
        total_wh = thermal_to_electrical_wh(thermal_wh, mean_cop)

        # With a COP per slot the solver can prefer the warm hours, and the demand is
        # then the electricity the heat would take at the *best* of them: each slot's
        # Wh counts at its own COP over that one, so the total is a statement about
        # heat rather than about any particular placement.
        efficiency, reference_cop = [], None
        if slot_efficiency.source_of(self.config) == slot_efficiency.SOURCE_COP:
            cops = self.cop_series(ctx)
            usable = [
                cops[index] for index in range(max(0, ctx.current_slot), ctx.slot_count)
                if feasible[index]
            ]
            if usable:
                reference_cop = max(usable)
                efficiency = slot_efficiency.normalise(
                    cops, ctx.slot_count, start=ctx.current_slot, feasible=feasible
                )
                total_wh = thermal_to_electrical_wh(thermal_wh, reference_cop)

        circuit_off, circuit_state = self.circuit_off(ctx)
        if circuit_off:
            # The controller decides, not the plan: nothing to place, nothing to
            # release, and no frost run either - a switched-off circuit cannot deliver
            # one. The energy figures stay, so the card still says what is missing.
            feasible = [False] * ctx.slot_count
            feasible_reason = [REASON_CIRCUIT_OFF] * ctx.slot_count
            urgent = False

        active, reason = self._control_state(current, target, ctx.readings, urgent)
        if not any(feasible):
            reason = REASON_OUT_OF_SEASON
        if circuit_off:
            active, reason = False, REASON_CIRCUIT_OFF

        store = None
        if (self.planning_mode == PLAN_BY_TEMPERATURE and self.volume_m3 > 0
                and not circuit_off):
            store = self.thermal_store(ctx, current, target, cover)

        power_w, power_curve = self.power_plan(ctx, current, target)
        power_now = (
            power_w[ctx.current_slot]
            if power_w and 0 <= ctx.current_slot < len(power_w) else self.rated_power_w
        )

        # The charge that has to happen: below the hard limit at once, below the soft
        # limit within the next allowed window, either way whatever the price limit
        # says. A store planned by temperature has its floor in the band instead.
        must_wh, must_slots, must_by = 0.0, [], None
        if store is None and not circuit_off and (urgent or below_soft):
            cop = reference_cop or mean_cop
            must_wh = min(total_wh, thermal_to_electrical_wh(heat_up_wh, cop))
            feasible, feasible_reason, must_slots, must_by = self._must_run(
                ctx, feasible, feasible_reason, must_wh, urgent, power_now
            )
            if not must_slots:
                must_wh = 0.0
            elif not urgent:
                # The soft limit sits below the deadband as a rule, but where it does
                # not, being under it is reason enough to start.
                active, reason = True, REASON_SOFT_LIMIT

        return EnergyDemand(
            kind=self.kind,
            total_wh=total_wh,
            feasible=feasible,
            feasible_reason=feasible_reason,
            max_power_w=self.rated_power_w,
            power_w=power_w,
            power_curve=power_curve,
            efficiency=efficiency,
            store=store,
            deadline_slot=deadline_slot,
            urgent=urgent and active,
            must_wh=must_wh,
            must_slots=must_slots,
            reason=reason,
            detail={
                "active": active,
                # The charge the limits demand: how much, and by when.
                "soft_limit_c": self.soft_limit_temp_c,
                "hard_limit_c": (
                    _as_float(self.frost_protection_temp_c)
                    if self.frost_protection_temp_c not in (None, "") else None
                ),
                "must_wh": round(must_wh, 1),
                # Whether the store counts as under its soft limit - it does until it
                # is the hysteresis above it again.
                "below_soft_limit": bool(below_soft),
                "limit_hysteresis_k": self.limit_hysteresis_k,
                "must_by": must_by.isoformat() if must_by is not None else None,
                # Whether a solver following the temperature may run it now: anywhere
                # below the storage ceiling, because inside the band the plan - not a
                # deadband - decides when heat is worth putting in.
                "storage_active": bool(
                    not circuit_off
                    and (urgent or current < target + self.storage_headroom_k)
                ),
                # The heat pump controller's own on/off for this circuit, where a
                # circuit_sensor is configured; None without one.
                "circuit_off": circuit_off,
                "circuit_state": circuit_state,
                "planning_mode": self.planning_mode,
                "efficiency_source": slot_efficiency.source_of(self.config),
                "reference_cop": _round_or_none(reference_cop, 2),
                "temperature_c": round(current, 2),
                # Seconds since the reading above was taken, when the sensor is
                # momentarily unreadable and the last value stands in; None when live.
                "temperature_held_s": None if held_s is None else round(held_s),
                "target_temperature_c": round(target, 2),
                "heat_up_wh_thermal": round(heat_up_wh, 1),
                "standing_losses_wh_thermal": round(losses_wh, 1),
                "mean_cop": round(mean_cop, 2),
                "cover_factor": cover_now,
                "cover_habit_hours": self.cover_habit.hours_known(),
                "running": self.is_running(ctx.readings),
                # The inputs, so a wrong answer can be diagnosed from the page. The
                # placeholder-ambient bug produced entirely plausible outputs and was
                # invisible precisely because none of this was reported.
                "ambient_now_c": (
                    round(self._ambient_at(ctx, ctx.current_slot), 1)
                    if self._ambient_at(ctx, ctx.current_slot) is not None else None
                ),
                "ambient_source": ctx.ambient_source,
                # What the site's own thermometer says, alongside what the model used.
                # "Outside now" promised a measurement and showed a forecast.
                "ambient_measured_c": _round_or_none(ctx.ambient_measured_c),
                "ambient_forecast_c": _round_or_none(ctx.ambient_forecast_c),
                "ambient_offset_k": _round_or_none(
                    None if (ctx.ambient_measured_c is None
                             or ctx.ambient_forecast_c is None)
                    else self._ambient_at(ctx, ctx.current_slot) - ctx.ambient_forecast_c
                ),
                # The heat pump's own air, where it differs from the surroundings.
                "cop_air_now_c": (
                    _round_or_none(self._cop_air_at(ctx, ctx.current_slot))
                    if getattr(ctx, "cop_air_temp_c", None) else None
                ),
                "cop_air_measured_c": _round_or_none(
                    getattr(ctx, "cop_air_measured_c", None)
                ),
                "horizon_hours": round(
                    max(0, ctx.slot_count - ctx.current_slot) * ctx.hours_per_slot(), 1
                ),
                # What it draws while running now: the power curve's figure for this
                # slot where it has one, its rating otherwise.
                "electrical_power_w": round(power_now, 1),
                "thermal_power_w": round(
                    power_now * mean_cop
                    * (factor_at(self.modulation, power_now)
                       if self.modulation and power_curve else 1.0)
                ),
                "power_curve": (
                    {**self.power_learner.state(), "power_now_w": round(power_now, 1)}
                    if power_curve else None
                ),
            },
        )

    def power_plan(self, ctx, current, target):
        """
        Per slot, what the appliance draws while running, and the formula behind it.

        ``([], None)`` for an appliance at its rated power. With a power curve every
        slot gets the curve's figure at that slot's air and at a first guess of the
        store's temperature - where it is now, settling back to target over the next
        hours. A solver following the store replaces the guess with the temperature it
        plans, which is why the formula goes along too: the base per slot (everything
        but the store term, the air already in it), the store term, the bounds, and
        the modulation COP factor across the range. Lists are in EOS slot space.
        """
        curve = self.power_curve()
        if curve is None:
            return [], None
        hours = ctx.hours_per_slot()
        airs, powers = [], []
        for index in range(ctx.slot_count):
            air = self._cop_air_at(ctx, index)
            air = float(air) if air is not None else 0.0
            ahead = max(0, index - ctx.current_slot) * hours
            airs.append(air)
            powers.append(round(curve.at(air, guess_store_c(current, target, ahead)), 1))
        solver = {
            "base_w": [round(curve.base_w + curve.air_w_per_k * air, 1) for air in airs],
            "store_w_per_k": round(curve.store_w_per_k, 3),
            "ref_c": round(curve.ref_c, 3),
            "min_w": round(curve.min_w, 1),
            "max_w": round(curve.max_w, 1),
            "cop_factor": factor_table(self.modulation, curve.min_w, curve.max_w),
        }
        return powers, solver

    def thermal_store(self, ctx, current, target, cover):
        """
        The store as a solver needs it to follow the temperature slot by slot.

        The same physics as the demand and the projection - losses against the air
        through the calibrated coefficient and the cover, heat in through the COP at
        each slot's air temperature, the sun through the fitted gain - laid out per
        slot instead of summed, so the three cannot disagree. The band is the deadband
        below target and the storage headroom above it: comfort at the bottom, what may
        be banked at the top. Lists are in EOS slot space, from local midnight.
        """
        solar = ctx.solar_wh or []
        gain = self.calibrator.solar_gain
        loss, ambient, gains = [], [], []
        for index in range(ctx.slot_count):
            air = self._ambient_at(ctx, index)
            ambient.append(air if air is not None else 0.0)
            factor = cover[index] if isinstance(cover, list) else cover
            loss.append(self.calibrator.loss_coefficient * self.surface_m2 * factor)
            gains.append(
                gain * max(0.0, float(solar[index]))
                if gain > 0 and index < len(solar) else 0.0
            )
        return {
            "capacity_wh_per_k": WH_PER_M3_PER_K * self.volume_m3,
            "loss_w_per_k": loss,
            "ambient_c": ambient,
            "cop": self.cop_series(ctx),
            "gain_wh": gains,
            "t_initial": float(current),
            "t_min": float(target) - self.deadband_k,
            "t_target": float(target),
            "t_max": float(target) + self.storage_headroom_k,
        }

    def cop_series(self, ctx):
        """The COP the appliance would run at in every slot, from its air temperature."""
        series = []
        for index in range(ctx.slot_count):
            air = self._cop_air_at(ctx, index)
            series.append(
                cop_at(air, self.calibrator.cop_nominal, self.calibrator.air_coefficient)
                if air is not None else self.calibrator.cop_nominal
            )
        return series

    def cover_series(self, ctx, measured):
        """
        Effective cover factor for every slot of the horizon.

        The next couple of hours take the switch as it reads now; beyond that the
        learned habit, blended rather than switched - an hour that is covered four
        nights in five is worth four fifths of a cover, and saying so is more honest
        than rounding it either way. An hour with too little history defers to the
        switch.
        """
        if self.cover_loss_factor >= 1.0:
            return [1.0] * ctx.slot_count

        observed_until = ctx.current_slot + round(
            OBSERVED_HOURS * 3600 / ctx.time_frame_base
        )

        series = []
        for index in range(ctx.slot_count):
            if index <= observed_until:
                series.append(measured)
                continue
            # The hour this slot actually starts at - see _slot_mask: index
            # arithmetic is an hour out after a DST transition, which would
            # read the cover habit for the wrong hour of the evening.
            hour = (
                ctx.anchor + timedelta(seconds=index * ctx.time_frame_base)
            ).astimezone(ctx.anchor.tzinfo).hour
            probability = self.cover_habit.probability(hour)
            if probability is None:
                series.append(measured)
            else:
                series.append(1.0 - probability * (1.0 - self.effective_cover_factor()))
        return series

    def _horizon_losses(self, ctx, target, cover, feasible, until=None):
        """
        Thermal energy lost between now and the end of the horizon, and the COP to use.

        Losses are evaluated at the *target* temperature: that is the state the plan is
        working to hold, and using the current temperature would understate the demand of
        a store that is still cold. The mean COP is taken over the slots the appliance may
        actually run in, because those are the ambient temperatures it will see.

        Sunshine is netted off. A store that gains heat from the sun needs less from the
        appliance, and a model that measured that gain without ever spending it would
        book the saving and then ask for the energy anyway. The gain is zero until the
        calibration has both bright and dark windows to separate it from everything else.
        """
        hours = ctx.hours_per_slot()
        solar = ctx.solar_wh or []
        gain = self.calibrator.solar_gain
        # A store with a deadline is asked to be at target *by* then, not held there
        # for two days. Integrating to the end of the horizon regardless asked a sauna
        # for 177 kWh of standing loss - 29 hours of running in a 42 hour horizon -
        # when what it wanted was to be hot in six.
        last = ctx.slot_count if until is None else min(ctx.slot_count, until + 1)

        losses = 0.0
        cops = []
        for index in range(max(0, ctx.current_slot), last):
            ambient = self._ambient_at(ctx, index)
            if ambient is None:
                continue
            losses += loss_power_w(
                self.calibrator.loss_coefficient, self.surface_m2, target, ambient,
                cover[index] if isinstance(cover, list) else cover,
            ) * hours
            if gain > 0 and index < len(solar):
                # solar_wh is energy per slot; the gain was fitted against power.
                losses -= gain * max(0.0, float(solar[index]))
            if feasible[index]:
                cops.append(
                    cop_at(self._cop_air_at(ctx, index), self.calibrator.cop_nominal,
                           self.calibrator.air_coefficient)
                )

        mean_cop = sum(cops) / len(cops) if cops else self.calibrator.cop_nominal
        return losses, mean_cop

    def project_medium(self, ctx, plan, start_c, target=None):
        """
        Where the store's own temperature goes, slot by slot, under a given plan.

        The card needs this to answer "why does it want so much energy?". The demand
        figure is a single number and says nothing about the shape behind it: a store
        that has to climb four kelvin *and* hold there against a cold night is asking
        for two different things at once, and only a curve separates them.

        It is the same physics the demand is built from, integrated rather than
        summed - heat in from whatever the plan runs, heat out to the air it is
        standing in, sun netted off - so the line and the number cannot disagree.
        Slots before now are the past and stay empty.
        """
        if self.volume_m3 <= 0:
            return []
        if target is None:
            target = self.target_temperature(ctx.readings)
            # A store planned by temperature is meant to go above target when heat is
            # cheap; stopping the line there would draw a plan that is not the one made.
            if self.planning_mode == PLAN_BY_TEMPERATURE:
                target += self.storage_headroom_k
        hours = ctx.hours_per_slot()
        cover = self.cover_series(ctx, self.cover_factor(ctx.readings))
        solar = ctx.solar_wh or []
        gain = self.calibrator.solar_gain
        capacity_wh_per_k = WH_PER_M3_PER_K * self.volume_m3

        temperature = float(start_c)
        projected = []
        for index in range(ctx.slot_count):
            if index < ctx.current_slot:
                projected.append(None)
                continue
            ambient = self._ambient_at(ctx, index)
            if ambient is None:
                projected.append(round(temperature, 2))
                continue

            electrical_wh = float(plan[index]) if index < len(plan) else 0.0
            gained = electrical_wh * cop_at(
                self._cop_air_at(ctx, index), self.calibrator.cop_nominal,
                self.calibrator.air_coefficient,
            )
            if electrical_wh > 0 and hours > 0 and self._modulation_factor() is not None:
                # The COP the planner gave this slot: scaled by the factor at its power.
                gained *= factor_at(self.modulation, electrical_wh / hours)
            gained -= loss_power_w(
                self.calibrator.loss_coefficient, self.surface_m2, temperature, ambient,
                cover[index] if isinstance(cover, list) else cover,
            ) * hours
            if gain > 0 and index < len(solar):
                gained += gain * max(0.0, float(solar[index]))

            temperature += gained / capacity_wh_per_k
            # The appliance stops at target; a projection that does not says a sauna
            # will reach 120 C against a 90 C setting. Invisible on a pool at four
            # hundredths of a kelvin per slot, plain on a 6 kW heater in 80 litres.
            if target is not None and temperature > target:
                temperature = float(target)
            projected.append(round(temperature, 2))
        return projected

    def _control_state(self, current, target, readings, urgent):
        """The hysteresis: when may the appliance start, and when must it stop."""
        if urgent:
            return True, REASON_FROST
        if current >= target:
            return False, REASON_AT_TARGET
        if current < target - self.deadband_k:
            return True, REASON_HEATING
        # Inside the deadband: keep going if already running, do not start otherwise.
        if self.is_running(readings):
            return True, REASON_HOLDING
        return False, REASON_IN_DEADBAND

    def _must_run(self, ctx, feasible, feasible_reason, must_wh, urgent, power_w):
        """
        Where the charge the limits demand has to land.

        Returns ``(feasible, feasible_reason, must_slots, must_by)``: the mask and its
        reasons (opened where the hard limit needs it), a per-slot mask of the slots
        the charge counts in, and the end of the last of them.

        Below the hard limit the store is charged now: the next run's worth of slots
        from the current one, with the window, season and air limit set aside for
        them - a tank at 30 C is not left until tomorrow's window. Below the soft limit
        the charge belongs in the next allowed window, the one running now or the first
        one ahead, and ``soft_limit_hours`` can pull its end in. Inside those slots the
        plan still picks the cheapest; only the price limit stops counting.
        """
        start = max(0, ctx.current_slot)
        count = ctx.slot_count
        if start >= count or must_wh <= 0:
            return feasible, feasible_reason, [], None
        feasible = list(feasible) + [True] * max(0, count - len(feasible))
        feasible_reason = (
            list(feasible_reason or []) + [None] * max(0, count - len(feasible_reason or []))
        )
        slot_wh = max(0.0, float(power_w or 0.0)) * ctx.hours_per_slot()
        run_slots = (
            max(1, int(math.ceil(must_wh / slot_wh))) if slot_wh > 0 else 1
        )

        if urgent:
            last = min(count - 1, start + max(run_slots, ctx.slots_per_hour()))
            for index in range(start, last + 1):
                feasible[index] = True
                feasible_reason[index] = None
        else:
            first = next(
                (index for index in range(start, count) if feasible[index]), None
            )
            if first is None:
                return feasible, feasible_reason, [], None
            last = first
            while last + 1 < count and feasible[last + 1]:
                last += 1
            if self.soft_limit_hours > 0:
                limit = start + int(
                    math.ceil(self.soft_limit_hours * ctx.slots_per_hour())
                ) - 1
                last = min(last, max(limit, first + run_slots - 1))

        must_slots = [
            start <= index <= last and bool(feasible[index]) for index in range(count)
        ]
        must_by = ctx.anchor + timedelta(seconds=(last + 1) * ctx.time_frame_base)
        return feasible, feasible_reason, must_slots, must_by

    # -- calibration ----------------------------------------------------------------------

    def observe(self, sample):
        """Hand one recorded sample to the calibrator, and to the cover habit."""
        if not isinstance(sample, dict) or "timestamp" not in sample:
            return False
        self._observe_cover(sample)
        learned = self.power_learner.observe(sample) if self.power_curve_active else False
        return bool(self.calibrator.observe(sample) or learned)

    def _observe_cover(self, sample):
        """Note whether the store was covered, for the per-hour pattern."""
        if self.cover_loss_factor >= 1.0:
            return
        covered = sample.get("covered")
        if covered is None:
            factor = sample.get("cover_factor")
            if factor is None:
                return
            try:
                covered = float(factor) < 1.0
            except (TypeError, ValueError):
                return
        self.cover_habit.observe(sample["timestamp"], bool(covered))

    def observe_history(self, samples):
        """Replay a recorded history at startup so day one is not a cold start."""
        rows = list(samples)
        for sample in rows:
            if isinstance(sample, dict):
                self._observe_cover(sample)
        if self.power_curve_active:
            self.power_learner.observe_series(rows)
        return self.calibrator.observe_series(rows)

    def status(self):
        state = self.calibrator.state()
        state.update({
            "cover_habit": self.cover_habit.state(),
            "volume_m3": self.volume_m3,
            "surface_m2": self.surface_m2,
            "rated_power_w": self.rated_power_w,
            "power_curve": (
                self.power_learner.state() if self.power_curve_active else None
            ),
            "deadband_k": self.deadband_k,
            "standby_threshold_w": self.standby_threshold_w,
        })
        return state
