"""
Turning a plan into a released / not released signal.

The plan says which slots the load should run in. That is not yet a safe control output:
a plan is recomputed every optimizer cycle, and a slot that was cheapest at 12:00 may not
be at 12:15. Acting on that directly would toggle a compressor several times an hour,
which is how you destroy a heat pump while saving four cents.

So the gate sits between the plan and the outside world and owns the things a plan has no
concept of - how long a release has to be honoured once given, and a manual override that
outranks both.
"""

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("__main__")


def _parse_time(value):
    """An ISO timestamp from the journal, aware; None when it is not one."""
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed

STATE_RELEASED = "released"
STATE_BLOCKED = "blocked"

OVERRIDE_RELEASE = "release"
OVERRIDE_BLOCK = "block"


class OverrideNotSupported(ValueError):
    """
    Asked to override a load that has no release signal.

    Its own type, so the route can answer it precisely instead of echoing whatever
    ValueError happened to surface - applying an override re-plans, and everything
    that runs underneath is entitled to raise one.
    """

# Reasons, surfaced verbatim on MQTT and the dashboard so a user can always answer
# "why is it running right now".
REASON_OVERRIDE = "manual override"
REASON_URGENT = "urgent - frost protection or missed deadline"
REASON_MIN_RUNTIME = "holding minimum runtime"
REASON_PLANNED = "planned cheap slot"
REASON_NOT_PLANNED = "not in a planned slot"
REASON_NO_DEMAND = "no demand"


class ReleaseGate:
    """
    Release decision for one managed load.

    The gate is deliberately stateful: whether the load may stop depends on when it
    started, which nothing else in the pipeline remembers.
    """

    def __init__(self, entry_id, min_runtime_minutes=0):
        self.id = entry_id
        self.min_runtime_minutes = max(0, int(min_runtime_minutes or 0))
        self.released = False
        self.released_since = None
        # The mirror of released_since. A plan is rebuilt from scratch every couple of
        # minutes, so without knowing how long the appliance has been *off* nothing can
        # stop the next plan restarting it moments after the last one stopped it.
        self.blocked_since = None
        self._override_mode = None
        self._override_until = None
        self._last_reason = REASON_NO_DEMAND

    # -- override ---------------------------------------------------------------------

    def set_override(self, mode, minutes, now):
        """
        Force release or block for a while. ``mode=None`` clears it.

        An override always wins, including over the minimum runtime: it exists for the
        moments when the user knows something the model does not, and an override the
        model can veto is not an override.
        """
        if mode is None:
            self._override_mode = None
            self._override_until = None
            logger.info("[LOADS] '%s' override cleared", self.id)
            return
        if mode not in (OVERRIDE_RELEASE, OVERRIDE_BLOCK):
            raise ValueError(f"unknown override mode {mode!r}")
        span = max(1, int(minutes or 1))
        self._override_mode = mode
        self._override_until = now + timedelta(minutes=span)
        logger.info(
            "[LOADS] '%s' override set to '%s' for %d minutes", self.id, mode, span
        )

    def override_active(self, now):
        """True while an override is in force; expires itself on read."""
        if self._override_mode is None:
            return False
        if now >= self._override_until:
            logger.info("[LOADS] '%s' override expired", self.id)
            self._override_mode = None
            self._override_until = None
            return False
        return True

    # -- decision ---------------------------------------------------------------------

    def evaluate(self, planned, urgent, now, has_demand=True, idle_reason=None):
        """
        Decide whether the load is released right now.

        Args:
            planned: True when the current slot carries planned energy.
            urgent: True when price no longer matters.
            now: Aware local time.
            has_demand: False once the target is reached - it ends a minimum-runtime
                hold, because holding a pump on against a satisfied setpoint would
                overshoot the very target the hold exists to reach.
            idle_reason: What to report instead of "no demand" when the model knows
                a more specific cause - a heating circuit its controller switched off.

        Returns a dict with ``released``, ``state`` and ``reason``.
        """
        if self.override_active(now):
            released = self._override_mode == OVERRIDE_RELEASE
            return self._settle(released, REASON_OVERRIDE, now)

        if urgent and has_demand:
            return self._settle(True, REASON_URGENT, now)

        if self.released and has_demand and self._within_min_runtime(now):
            return self._settle(True, REASON_MIN_RUNTIME, now)

        if not has_demand:
            return self._settle(False, idle_reason or REASON_NO_DEMAND, now)

        if planned:
            return self._settle(True, REASON_PLANNED, now)

        return self._settle(False, REASON_NOT_PLANNED, now)

    def _within_min_runtime(self, now):
        if not self.min_runtime_minutes or self.released_since is None:
            return False
        elapsed = (now - self.released_since).total_seconds() / 60.0
        return elapsed < self.min_runtime_minutes

    def _settle(self, released, reason, now):
        if released and not self.released:
            self.released_since = now
            self.blocked_since = None
            logger.info("[LOADS] '%s' released (%s)", self.id, reason)
        elif not released and self.released:
            held = 0.0
            if self.released_since is not None:
                held = (now - self.released_since).total_seconds() / 60.0
            self.released_since = None
            self.blocked_since = now
            logger.info(
                "[LOADS] '%s' blocked after %.0f minutes (%s)", self.id, held, reason
            )

        self.released = released
        self._last_reason = reason
        return {
            "released": released,
            "state": STATE_RELEASED if released else STATE_BLOCKED,
            "reason": reason,
            "released_since": self.released_since.isoformat() if self.released_since else None,
            "override": self._override_mode,
            "override_until": self._override_until.isoformat() if self._override_until else None,
        }

    # -- after a restart ---------------------------------------------------------------

    def restore(self, decisions, now, max_age_s):
        """
        Pick up where the gate stood before EOS Connect was restarted.

        *decisions* is the journal, newest first, each with an ISO ``timestamp`` and
        ``released``. The newest one says whether the load was released; the unbroken
        run of equal decisions behind it says since when - so a run that was 40 minutes
        into a 90-minute minimum keeps the remaining 50, and a rest keeps its rest. A
        journal older than *max_age_s* describes a past nobody is still in and is
        ignored.

        Without this every restart opened with a blocked gate and no history: a heat
        pump running at full power read as stopped, so continuing it was priced as a
        fresh start and the first plan stopped it - 26.11., 21:34, 50 minutes into a
        run that was planned until midnight, restarted an hour later.

        Returns True when a state was restored.
        """
        if now.tzinfo is None:
            now = now.astimezone()
        entries = []
        for row in decisions or []:
            moment = _parse_time(row.get("timestamp"))
            if moment is not None:
                entries.append((moment, bool(row.get("released"))))
        if not entries:
            return False
        entries.sort(key=lambda entry: entry[0], reverse=True)
        newest, released = entries[0]
        if (now - newest).total_seconds() > max_age_s:
            return False

        since = newest
        for moment, state in entries[1:]:
            # A hole in the journal is a time nobody was deciding: the run before it
            # is not the one still going.
            if state != released or (since - moment).total_seconds() > max_age_s:
                break
            since = moment
        self.released = released
        self.released_since = since if released else None
        self.blocked_since = None if released else since
        self._last_reason = REASON_PLANNED if released else REASON_NOT_PLANNED
        logger.info(
            "[LOADS] '%s' restored as %s since %s from its journal",
            self.id, STATE_RELEASED if released else STATE_BLOCKED,
            since.isoformat(timespec="minutes"),
        )
        return True

    def adopt_running(self, power_w, threshold_w):
        """
        Take an appliance that is drawing power as running, when nothing better is known.

        The journal's fallback: a load restored from nothing - a fresh database, an
        outage longer than the journal is trusted for - but measurably on. Released with
        no start time, so continuing costs no start while no minimum run is invented;
        the next plan still decides whether it stops.
        """
        if self.released or power_w is None or power_w <= threshold_w:
            return False
        self.released = True
        self.released_since = None
        self.blocked_since = None
        self._last_reason = REASON_PLANNED
        logger.info(
            "[LOADS] '%s' is drawing %.0f W at startup - taken as running", self.id,
            power_w,
        )
        return True

    def status(self, now=None):
        """Current gate state without changing it."""
        return {
            "released": self.released,
            "state": STATE_RELEASED if self.released else STATE_BLOCKED,
            "reason": self._last_reason,
            "released_since": self.released_since.isoformat() if self.released_since else None,
            "min_runtime_minutes": self.min_runtime_minutes,
            "override": self._override_mode,
            "override_until": self._override_until.isoformat() if self._override_until else None,
        }
