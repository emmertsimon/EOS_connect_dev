"""
Bundled MILP optimizer engine — adapted from evcc-io/optimizer.

Original source: https://github.com/evcc-io/optimizer
License: MIT License
Copyright (c) 2025 andig

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---
Modifications made for EOS_connect integration (adapted from main branch, ~2025-06):
- Removed Flask/flask-restx/pydantic dependencies; OptimizerSettings is now a plain dataclass
- Added 'maximize_self_consumption' charging strategy
- Added 'emergency_reserve' discharging strategy (end-of-horizon SOC floor)
- Added 'emergency_reserve' fields (s_reserve) to BatteryConfig and corresponding
  penalty variable/constraint in the MILP model
- Added gapRel to OptimizerSettings and the CBC invocation (1% optimality gap)
- Solver resolution probes CBC candidates with a real execve() and prefers the
  first that runs (system PATH, then pulp's bundled binary), because pulp's
  bundled x86_64 CBC is glibc-linked and cannot exec on Alpine/musl
- Per-slot tight Big-M bounds in energy-balance and battery constraints replace the
  upstream global M=1e6, tightening the LP relaxation and reducing B&B tree size
- or 0.0 guard on pulp.value() calls in solve() to handle None results
- get_clean_objective_value() measures the battery term from bat.s_initial
  rather than s[0]; upstream's s[0] is the SOC *after* the first slot has
  charged or discharged, so the first slot fell out of the reported delta
- Module is invoked in-process; no HTTP server needed
- Added ManagedLoadConfig and the variables, constraints and objective term that let
  the solver *place* a contingent load (pool heat pump, sauna, hot-water tank) rather
  than being handed it as fixed demand, so the battery and the load are decided together
"""

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pulp

# eos_connect.py attaches its handlers to the "__main__" logger, not to root, so
# a logging.getLogger(__name__) here propagates to a handler-less root and the
# solver diagnostics are silently discarded.
logger = logging.getLogger("__main__")

# Seconds allowed for a CBC candidate to answer `cbc -quit`.
CBC_PROBE_TIMEOUT_S = 15.0

# Cached probe result: the CBC path to use, or None until probed. The probe
# forks a subprocess and solves run on a timer, so it must happen only once.
_resolved_cbc: Optional[str] = None


class CbcSolverUnavailableError(RuntimeError):
    """No runnable CBC binary could be found for the built-in optimizer."""


def _cbc_runs(path: str) -> Optional[str]:
    """
    Return None if `path` can actually be executed, else a rejection reason.

    Executing the candidate is required, not optional: pulp's
    COIN_CMD.available() only does os.path.exists() + os.access(X_OK), so on
    Alpine/musl the glibc-linked CBC bundled in the pulp wheel passes that
    check and then fails at solve time with a bare
    "FileNotFoundError: .../solverdir/cbc/linux/i64/cbc" -- the file exists,
    but its ELF interpreter /lib64/ld-linux-x86-64.so.2 does not.
    """
    try:
        proc = subprocess.run(
            [path, "-quit"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=CBC_PROBE_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError:
        return (
            f"{path}: exists but cannot be started - missing ELF interpreter "
            "(a glibc-linked binary on a musl system, e.g. Alpine Linux / "
            "Home Assistant OS add-on)"
        )
    except OSError as exc:
        return f"{path}: could not be started ({exc})"
    except subprocess.TimeoutExpired:
        return f"{path}: no response within {CBC_PROBE_TIMEOUT_S:.0f}s"
    if proc.returncode < 0:
        return (
            f"{path}: killed by signal {-proc.returncode} "
            "(signal 4 = illegal instruction: the binary needs CPU features "
            "this machine does not provide, e.g. Proxmox kvm64/qemu64)"
        )
    if proc.returncode != 0:
        return f"{path}: exited with status {proc.returncode}"
    return None


def _bundled_cbc_path() -> Optional[str]:
    """Path of the CBC binary shipped inside the installed pulp wheel, if any."""
    cls = getattr(pulp, "PULP_CBC_CMD", None)  # removed in PuLP 4.0
    if cls is None:
        return None
    # Read the class attribute rather than instantiating, to avoid PuLP 3.x's
    # "PULP_CBC_CMD will be removed in PuLP 4.0" deprecation warning.
    return getattr(cls, "pulp_cbc_path", None)


def probe_cbc_solver() -> str:
    """
    Determine once which CBC binary to use, logging the decision and any
    rejections, and return its path.

    Preference order:
      1. a system `cbc` on PATH  -- lets an image ship a working build
      2. the CBC bundled with pulp -- the historical default; fine on glibc

    Both candidates must survive a real execve() probe, so a broken system
    `cbc` no longer shadows a working bundled binary.
    """
    global _resolved_cbc  # pylint: disable=global-statement
    if _resolved_cbc is not None:
        return _resolved_cbc

    rejections = []
    candidates = (
        ("system", shutil.which("cbc")),
        ("bundled", _bundled_cbc_path()),
    )
    for label, path in candidates:
        if not path:
            logger.debug("[OPT-LocalEVopt] No %s CBC binary found", label)
            continue
        reason = _cbc_runs(path)
        if reason is None:
            logger.info("[OPT-LocalEVopt] Using %s CBC solver: %s", label, path)
            _resolved_cbc = path
            return path
        rejections.append(f"{label} CBC -> {reason}")
        logger.warning("[OPT-LocalEVopt] Rejected %s CBC -- %s", label, reason)

    raise CbcSolverUnavailableError(
        "No runnable CBC solver found for the built-in optimizer (local_evopt).\n"
        + ("\n".join(f"  - {r}" for r in rejections) or "  - no CBC binary found at all")
        + "\nOn Home Assistant OS x86_64 add-ons the CBC binary bundled with pulp "
        "is glibc-linked and cannot run on Alpine/musl; update to an add-on "
        "version that ships a static CBC solver, or switch the optimization "
        "backend away from the built-in optimizer in Settings."
    )


def _resolve_cbc_solver(
    msg: int = 0,
    num_threads: Optional[int] = None,
    time_limit: Optional[float] = None,
    gapRel: Optional[float] = None,
) -> pulp.LpSolver:
    """
    Return a configured PuLP CBC solver bound to a verified-runnable binary.

    COIN_CMD is used for both candidates: it is PULP_CBC_CMD's own base class
    with an identical solve path, it accepts an explicit path, and it survives
    PuLP 4.0 dropping PULP_CBC_CMD.
    """
    return pulp.COIN_CMD(
        path=probe_cbc_solver(),
        msg=msg,
        threads=num_threads,
        timeLimit=time_limit,
        gapRel=gapRel,
    )


@dataclass
class OptimizerSettings:
    """Solver settings (replaces pydantic-based settings from upstream)."""
    num_threads: Optional[int] = None
    time_limit: Optional[float] = None
    # 1% optimality gap — negligible for energy, significantly faster
    gapRel: Optional[float] = 0.01


@dataclass
class OptimizationStrategy:
    """Optimization strategy settings for charging and discharging behavior."""
    charging_strategy: str = "none"
    discharging_strategy: str = "none"


@dataclass
class GridConfig:
    """Grid connection configuration including import/export limits and pricing."""
    p_max_imp: Optional[float] = None
    p_max_exp: Optional[float] = None
    prc_p_exc_imp: Optional[float] = None


@dataclass
class BatteryConfig:
    """Battery configuration including capacity, power limits, and optimization constraints."""
    s_min: float = 0.0
    s_max: float = 0.0
    s_initial: float = 0.0
    c_min: float = 0.0
    c_max: float = 0.0
    d_max: float = 0.0
    p_a: float = 0.0
    charge_from_grid: bool = False
    discharge_to_grid: bool = False
    s_capacity: Optional[float] = None
    p_demand: Optional[List[float]] = None
    s_goal: Optional[List[float]] = None
    c_priority: int = 0
    # Emergency reserve: minimum end-of-horizon SOC in Wh (EOS_connect extension)
    s_reserve: float = 0.0
    # SOC-dependent charge limit (EOS_connect extension), see ChargeCurveConfig.
    charge_curve: Optional["ChargeCurveConfig"] = None

    def __post_init__(self):
        if self.s_capacity is None:
            self.s_capacity = self.s_max


@dataclass
class ChargeCurveConfig:
    """
    The battery's charging curve in the solver's units (EOS_connect extension).

    The charge in a slot is held under the curve at the slot's average state of
    charge. Every line is ``(s_anchor_wh, power_w, slope_w_per_wh)`` and stands for
    ``power_w + slope_w_per_wh * (s_avg - s_anchor_wh)``.

    Attributes:
        pieces: The curve cut into concave stretches,
            ``(s_from_wh, s_to_wh, [line, ...])``; within a stretch the limit is the
            minimum of its lines. One piece means the curve is concave and the whole
            thing is linear.
        envelope: Lines of the smallest concave curve on or above the real one. On
            its own it keeps the problem linear and is exact up to
            ``envelope_exact_below``, generous above.
        envelope_exact_below: Wh of state of charge up to which the envelope equals
            the curve.
    """
    pieces: List[Tuple[float, float, List[Tuple[float, float, float]]]]
    envelope: List[Tuple[float, float, float]]
    envelope_exact_below: float


@dataclass
class ManagedLoadConfig:
    """
    A load whose *timing* is the optimizer's to choose (EOS_connect extension).

    The household profile `gt` is fixed: it says what the house will draw and the
    optimizer works around it. A pool heat pump, a sauna or a hot-water tank is not
    like that - it needs a quantity of energy by a deadline and does not care when,
    which makes it something the solver should be placing rather than being told.

    Placing it here rather than in `gt` is what lets the battery and the load be
    decided together: charging at 14 ct to run a pump through a 30 ct evening is a
    trade only something that sees both can make. Every attempt to make that call
    from outside had to guess what energy would cost, and the guess is what the
    solver is computing.

    Nothing here is specific to any appliance. What the energy is *for* lives in the
    demand model that produced these numbers.

    Attributes:
        id: Instance id, echoed back on the result so schedules can be matched up.
        demand_wh: Energy wanted over the horizon. An upper bound, not a target - a
            window too narrow to hold it must yield a short plan, not an infeasible
            model.
        value_eur_per_wh: What a Wh delivered here is worth. The solver runs the load
            in a slot exactly when the marginal cost of supplying it is below this, so
            it doubles as the price limit: the energy actually bought comes in under
            this figure by construction. Set it above any plausible tariff to mean
            "no limit".
        feasible: Per-slot mask - window, ambient temperature, season.
        max_power_w: Draw while running, which sets the per-slot ceiling.
        min_runtime_slots: Shortest run the appliance may be started for. Above 1 this
            costs a binary per slot.
        urgent_wh: Energy that must be placed whatever it costs - frost protection.
        committed_on_slots: Leading slots the appliance is already running through and
            may not be stopped in - the remainder of a minimum run begun under an
            earlier plan. Every plan is built from scratch, so without this the seam
            between two of them is where the appliance cycles.
        committed_off_slots: The mirror: leading slots it may not be restarted in.
        start_cost_eur: What one start is worth avoiding. Without it the objective is
            indifferent between a contiguous run and the same energy scattered over the
            day - every schedule of the same slots costs the same - so the solver
            returns whichever the search reaches first, and the appliance cycles.
    """

    id: str = ""
    demand_wh: float = 0.0
    value_eur_per_wh: float = 0.0
    feasible: Optional[List[bool]] = None
    max_power_w: float = 0.0
    min_runtime_slots: int = 1
    urgent_wh: float = 0.0
    start_cost_eur: float = 0.0
    # Cap on slots this load may run in one calendar day, 0 for none, with a label per
    # slot saying which day it belongs to. A label rather than a boundary because it
    # makes the same rotation into solver space as everything else indexed by slot;
    # -1 marks a slot that belongs to no day and is never capped.
    max_slots_per_day: int = 0
    day_index: list = None
    committed_on_slots: int = 0
    # Whether a run is under way. Separate from committed_on_slots on purpose: this
    # only tells the solver that continuing costs no start, where committing *forces*
    # the head slot on. Binding the two latches the load - the pin keeps the gate
    # released, which pins again next cycle - and it ran a live pool through an
    # evening peak it should have skipped.
    already_running: bool = False
    committed_off_slots: int = 0


@dataclass
class TimeSeriesData:
    """Time series input data for optimization including load, production, and prices."""
    dt: List[int]          # Time step length [s]
    gt: List[float]        # Required total energy [Wh]
    ft: List[float]        # Forecasted production [Wh]
    p_N: List[float]       # Import prices [currency unit/Wh]
    p_E: List[float]       # Export prices [currency unit/Wh]


class Optimizer:
    """
    MILP optimizer: builds the optimization model from input data and provides
    a solve() function to run optimization and return results.

    Supported charging_strategy values:
        'none'                     — no secondary preference
        'charge_before_export'     — prefer charging batteries before exporting (upstream)
        'attenuate_grid_peaks'     — charge at high solar yield times (upstream)
        'maximize_self_consumption'— penalize grid import when PV is available (EOS_connect)

    Supported discharging_strategy values:
        'none'                     — no secondary preference
        'discharge_before_import'  — prefer discharging batteries before grid import (upstream)
        'emergency_reserve'        — keep end-of-horizon SOC above s_reserve (EOS_connect)
    """

    def __init__(
        self,
        strategy: OptimizationStrategy,
        grid: GridConfig,
        batteries: List[BatteryConfig],
        time_series: TimeSeriesData,
        eta_c: float = 0.95,
        eta_d: float = 0.95,
        M: float = 1e6,
        optimizer_settings: Optional[OptimizerSettings] = None,
        managed_loads: Optional[List[ManagedLoadConfig]] = None,
        managed_load_budget_w: float = 0.0,
    ):
        self.settings = optimizer_settings or OptimizerSettings()
        self.strategy = strategy
        self.grid = grid
        self.batteries = batteries
        self.managed_loads = list(managed_loads or [])
        self.managed_load_budget_w = max(0.0, float(managed_load_budget_w or 0.0))
        self.time_series = time_series
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.M = M
        # number of time steps
        self.T = len(time_series.gt)
        # time step range
        self.time_steps = range(self.T)
        # the optimization problem
        self.problem = None
        # dictionary of optimizer variables
        self.variables: Dict[str, Any] = {}
        # Slots whose charging curve is held exactly, per battery (see solve()).
        self._curve_exact_slots: Dict[int, set] = {i: set() for i in range(len(batteries))}

        # Compute scaling for strategy control parameters
        self.min_import_price = np.min(self.time_series.p_N) if self.time_series.p_N else 0.0
        self.max_import_price = np.max(self.time_series.p_N) if self.time_series.p_N else 0.0

        # scaling for penalty parameters. Make sure goal_penalty is always positive
        # Use np.max() to floor penalty_base at 0.1e-3 — ensures penalties are
        # non-zero even when prices are zero (matches upstream evcc-io/optimizer).
        penalty_base = np.max([self.max_import_price, 0.1e-3])
        self.prc_e_goal_pen = penalty_base * 10e1
        self.prc_p_goal_pen = penalty_base * np.max(self.time_series.dt) / 3600 * 10e1
        self.prc_soc_exc_pen = penalty_base * 10e2

        # penalty for exceeding grid import limit
        self.prc_e_grid_imp_pen = penalty_base * 10e1
        # penalty for exceeding the grid export limit
        self.prc_e_grid_exp_pen = penalty_base * 10e1

        # demand rate flag
        self.is_grid_demand_rate_active = False
        if self.grid.p_max_imp is not None and self.grid.prc_p_exc_imp is not None:
            self.is_grid_demand_rate_active = True

    def create_model(self):
        """Create and initialize the MILP model."""
        self.problem = pulp.LpProblem("EV_Charging_Optimization", pulp.LpMaximize)
        self._setup_variables()
        self._setup_target_function()
        self._add_energy_balance_constraints()
        self._add_battery_constraints()
        self._add_managed_load_constraints()
        self._add_shared_budget_constraints()

    def _new_var(self, name, lowBound=None, upBound=None, cat=pulp.LpContinuous):
        # pylint: disable=invalid-name
        """A decision variable owned by this model.

        PuLP deprecated bare ``LpVariable(...)`` construction in favour of
        ``LpProblem.add_variable`` and warns once per variable - which, at a few
        thousand variables per run, drowns everything else in the test log. The new
        method only exists in recent PuLP and ``requirements.txt`` still allows 2.7,
        so fall back to the classic constructor when it is missing. Argument names
        follow PuLP's own spelling so the call sites read unchanged.
        """
        adder = getattr(self.problem, "add_variable", None)
        if adder is not None:
            return adder(name, lowBound, upBound, cat)
        return pulp.LpVariable(name, lowBound=lowBound, upBound=upBound, cat=cat)

    def _setup_variables(self):
        """Set up the variables of the MILP optimizer."""
        # Charging power variables [Wh]
        self.variables['c'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['c'][i] = [
                self._new_var(
                    f"c_{i}_{t}",
                    lowBound=0,
                    upBound=bat.c_max * self.time_series.dt[t] / 3600.
                )
                for t in self.time_steps
            ]

        # Discharging power variables [Wh]
        self.variables['d'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['d'][i] = [
                self._new_var(
                    f"d_{i}_{t}",
                    lowBound=0,
                    upBound=bat.d_max * self.time_series.dt[t] / 3600.
                )
                for t in self.time_steps
            ]

        # State of charge variables [Wh]
        self.variables['s'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['s'][i] = [
                self._new_var(f"s_{i}_{t}", lowBound=0, upBound=bat.s_capacity)
                for t in self.time_steps
            ]

        # penalty variable for not reaching given charge goals
        # variables are kept in a matrix Batteries X time steps
        self.variables['s_goal_pen'] = [
            [None for t in self.time_steps] for i in range(len(self.batteries))
        ]
        for i, bat in enumerate(self.batteries):
            if self.batteries[i].s_goal is not None:
                for t in self.time_steps:
                    if self.batteries[i].s_goal[t] > 0:
                        self.variables['s_goal_pen'][i][t] = self._new_var(
                            f"s_goal_pen_{i}_{t}", lowBound=0
                        )

        # penalty variable for not being able to charge with the required power
        self.variables['p_demand_pen'] = [
            [None for t in self.time_steps] for i in range(len(self.batteries))
        ]
        # binary variable to allow one out of two alternative constraints
        self.variables['z_p_demand'] = [
            [None for t in self.time_steps] for i in range(len(self.batteries))
        ]
        for i, bat in enumerate(self.batteries):
            if bat.p_demand is not None:
                for t in self.time_steps:
                    self.variables['p_demand_pen'][i][t] = self._new_var(
                        f"p_demand_pen_{i}_{t}", lowBound=0
                    )
                    self.variables['z_p_demand'][i][t] = self._new_var(
                        f"z_p_demand_{i}_{t}", cat='Binary'
                    )

        # penalty variable for staying above max SOC and below min SOC
        self.variables['s_max_pen'] = [
            [self._new_var(f"s_max_pen_{i}_{t}", lowBound=0) for t in self.time_steps]
            for i in range(len(self.batteries))
        ]
        self.variables['s_min_pen'] = [
            [self._new_var(f"s_min_pen_{i}_{t}", lowBound=0) for t in self.time_steps]
            for i in range(len(self.batteries))
        ]

        # Emergency reserve penalty variable (EOS_connect extension)
        # Penalizes end-of-horizon SOC below s_reserve
        self.variables['s_reserve_pen'] = [
            None for i in range(len(self.batteries))
        ]
        for i, bat in enumerate(self.batteries):
            if bat.s_reserve > 0:
                self.variables['s_reserve_pen'][i] = self._new_var(
                    f"s_reserve_pen_{i}", lowBound=0
                )

        # Grid import/export variables [Wh]
        self.variables['n'] = [self._new_var(f"n_{t}", lowBound=0) for t in self.time_steps]
        self.variables['e'] = [self._new_var(f"e_{t}", lowBound=0) for t in self.time_steps]

        # penalty variables for exceeding grid power limits (W)
        # for grid import
        if self.grid.p_max_imp is not None:
            self.variables['e_imp_lim_exc'] = [
                self._new_var(f"p_imp_pen_{t}", lowBound=0) for t in self.time_steps
            ]
            self.variables['z_imp_lim'] = [
                self._new_var(f"z_imp_lim_{t}", cat='Binary') for t in self.time_steps
            ]

        # for grid export
        if self.grid.p_max_exp is not None:
            self.variables['e_exp_lim_exc'] = [
                self._new_var(f"e_exp_lim_exc_{t}", lowBound=0) for t in self.time_steps
            ]
            self.variables['z_exp_lim'] = [
                self._new_var(f"z_exp_lim_{t}", cat='Binary') for t in self.time_steps
            ]

        # for demand rate calculation
        if self.is_grid_demand_rate_active:
            self.variables['p_max_imp_exc'] = self._new_var("p_max_imp_exc", lowBound=0)

        # Binary variable: power flow direction to / from grid
        self.variables['y'] = [
            self._new_var(f"y_{t}", cat='Binary') for t in self.time_steps
        ]

        # Binary variable for charging activation (only when c_min > 0)
        self.variables['z_c'] = {}
        for i, bat in enumerate(self.batteries):
            if bat.c_min > 0:
                self.variables['z_c'][i] = [
                    self._new_var(f"z_c_{i}_{t}", cat='Binary')
                    for t in self.time_steps
                ]
            else:
                self.variables['z_c'][i] = None

        # One per load: whether this slot begins a run. Only created when starts are
        # priced, since otherwise it is a binary the objective never reads.
        self.variables['ml_start'] = {}
        for i, load in enumerate(self.managed_loads):
            self.variables['ml_start'][i] = (
                [self._new_var(f"ml_start_{i}_{t}", cat='Binary')
                 for t in self.time_steps]
                if load.start_cost_eur > 0 else None
            )

        # Managed loads: energy placed in each slot, and whether the appliance is
        # running there. The mask is applied as an upper bound of zero rather than by
        # leaving the variable out, so every list stays T long and the result can be
        # read back positionally.
        self.variables['ml'] = {}
        self.variables['ml_on'] = {}
        for i, load in enumerate(self.managed_loads):
            per_slot = [
                load.max_power_w * self.time_series.dt[t] / 3600.0
                for t in self.time_steps
            ]
            feasible = load.feasible or [True] * self.T
            self.variables['ml'][i] = [
                self._new_var(
                    f"ml_{i}_{t}",
                    lowBound=0,
                    upBound=per_slot[t] if (
                        t < len(feasible) and feasible[t]
                    ) else 0.0,
                )
                for t in self.time_steps
            ]
            # Worth a binary when something actually reads the on/off state: a minimum
            # run to enforce, or starts being priced. Without either, the appliance may
            # be placed in any slot and the switch carries nothing the continuous
            # variable does not already have.
            if load.min_runtime_slots > 1 or load.start_cost_eur > 0:
                self.variables['ml_on'][i] = [
                    self._new_var(f"ml_on_{i}_{t}", cat='Binary')
                    for t in self.time_steps
                ]
            else:
                self.variables['ml_on'][i] = None

        # Binary variable to lock charging against discharging
        self.variables['z_cd'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['z_cd'][i] = [
                self._new_var(f"z_cd_{i}_{t}", cat='Binary')
                for t in self.time_steps
            ]

    def _setup_target_function(self):
        """Gather all target function contributions and instantiate the objective."""
        objective = 0

        # -----------------------------------------------------------------------
        # Primary cost & benefit elements
        # -----------------------------------------------------------------------

        # Grid import cost (negative → we want to minimize cost) [currency unit]
        for t in self.time_steps:
            if self.grid.p_max_imp is not None:
                objective -= (
                    self.variables['n'][t]
                    + self.variables['e_imp_lim_exc'][t]
                ) * self.time_series.p_N[t]
            else:
                objective -= self.variables['n'][t] * self.time_series.p_N[t]

        # Grid export revenue [currency unit]
        for t in self.time_steps:
            objective += self.variables['e'][t] * self.time_series.p_E[t]

        # Managed load value [currency unit]
        #
        # This single term is the whole price rule. The solver adds a Wh here only when
        # supplying it costs less than the Wh is worth - and "costs" is whatever the
        # cheapest source in that slot really is, grid import, forgone export, or a
        # battery it may charge earlier for exactly this purpose. So the energy bought
        # for the load comes in under `value_eur_per_wh` by construction; there is no
        # separate cap to enforce, and no averaging window to argue about.
        for i, load in enumerate(self.managed_loads):
            objective += load.value_eur_per_wh * pulp.lpSum(self.variables['ml'][i])
            starts = self.variables['ml_start'][i]
            if starts is not None:
                objective -= load.start_cost_eur * pulp.lpSum(starts)

        # Final state of charge value [currency unit]
        for i, bat in enumerate(self.batteries):
            objective += self.variables['s'][i][-1] * bat.p_a

        # Demand rate charge
        if self.is_grid_demand_rate_active:
            objective += -self.grid.prc_p_exc_imp * self.variables['p_max_imp_exc']

        # -----------------------------------------------------------------------
        # Penalties for exceeding battery SOC limits at start
        # -----------------------------------------------------------------------
        for i, bat in enumerate(self.batteries):
            for t in self.time_steps:
                objective += -self.prc_soc_exc_pen * (
                    self.variables['s_max_pen'][i][t] + self.variables['s_min_pen'][i][t]
                )

        # -----------------------------------------------------------------------
        # Penalties for goals that cannot be met
        # -----------------------------------------------------------------------
        for i, bat in enumerate(self.batteries):
            # unmet battery charging goals
            if self.batteries[i].s_goal is not None:
                for t in self.time_steps:
                    if self.batteries[i].s_goal[t] > 0:
                        objective += -self.prc_e_goal_pen * self.variables['s_goal_pen'][i][t]
            # unmet charging demand
            if bat.p_demand is not None:
                for t in self.time_steps:
                    objective += (
                        -self.prc_p_goal_pen
                        * self.variables['p_demand_pen'][i][t]
                        * (1 + (self.T - t) / self.T)
                    )

        # -----------------------------------------------------------------------
        # Penalties for grid power limits that cannot be met
        # -----------------------------------------------------------------------
        for t in self.time_steps:
            if self.grid.p_max_imp is not None and not self.is_grid_demand_rate_active:
                objective += -self.prc_e_grid_imp_pen * self.variables['e_imp_lim_exc'][t]
            if self.grid.p_max_exp is not None:
                objective += (
                    -self.prc_e_grid_exp_pen
                    * (1.0 - t * 1e-5)
                    * self.variables['e_exp_lim_exc'][t]
                )

        # -----------------------------------------------------------------------
        # Emergency reserve penalty (EOS_connect extension)
        # Strongly penalize ending below s_reserve — applied to all batteries
        # that have s_reserve > 0.  Uses a large penalty to make the reserve a
        # near-hard constraint while keeping the problem always feasible.
        # -----------------------------------------------------------------------
        for i, bat in enumerate(self.batteries):
            if bat.s_reserve > 0 and self.variables['s_reserve_pen'][i] is not None:
                # penalty weight: 1000x the goal penalty to make it near-hard
                prc_reserve = self.prc_e_goal_pen * 1000
                objective += -prc_reserve * self.variables['s_reserve_pen'][i]

        # -----------------------------------------------------------------------
        # Secondary strategies (cost-neutral preferences, small weights)
        # -----------------------------------------------------------------------

        # charge_before_export: prefer charging first, then export
        if self.strategy.charging_strategy == 'charge_before_export':
            for i, bat in enumerate(self.batteries):
                for t in self.time_steps:
                    objective += (
                        -self.variables['e'][t]
                        * self.min_import_price
                        * 2e-5
                        * (self.T - t)
                    )

        # attenuate_grid_peaks: charge at high solar production times
        if self.strategy.charging_strategy == 'attenuate_grid_peaks':
            for i, bat in enumerate(self.batteries):
                for t in self.time_steps:
                    objective += (
                        self.variables['c'][i][t]
                        * self.time_series.ft[t]
                        * self.min_import_price
                        * 1e-6
                    )

        # maximize_self_consumption (EOS_connect):
        # Prefer using PV locally over feeding it to the grid, even at a small
        # economic cost.  Unlike charge_before_export (which is a near-invisible
        # tie-breaker), this strategy uses a weight proportional to the feed-in
        # tariff (~15 %) so the optimizer will charge from PV even when the round-
        # trip economics are only marginally in favour of exporting.
        #
        # Effect on break-even: lowers the future-import-price threshold for
        # charging from ~p_E/η_rt  (pure economics) to a lower value, meaning the
        # battery is filled from PV more aggressively.
        if self.strategy.charging_strategy == 'maximize_self_consumption':
            # sc_weight ≈ 15 % of average feed-in tariff — visible preference but
            # still allows clear arbitrage to dominate when spreads are large.
            avg_feedin = float(np.mean(self.time_series.p_E)) if self.time_series.p_E else 0.0
            sc_weight = avg_feedin * 0.15
            for t in self.time_steps:
                if self.time_series.ft[t] > 0:
                    # Penalise exporting during PV production hours
                    objective += -self.variables['e'][t] * sc_weight
                    # Reward charging during PV production hours
                    for i, bat in enumerate(self.batteries):
                        objective += self.variables['c'][i][t] * sc_weight * 0.5

        # discharge_before_import: prefer discharging batteries before importing
        if self.strategy.discharging_strategy == 'discharge_before_import':
            for i, bat in enumerate(self.batteries):
                for t in self.time_steps:
                    objective += (
                        -self.variables['n'][t]
                        * self.min_import_price
                        * 5e-6
                        * (self.T - t)
                    )

        # charging and discharging priorities
        for i, bat in enumerate(self.batteries):
            for t in self.time_steps:
                objective += (
                    self.variables['c'][i][t]
                    * self.min_import_price
                    * 5e-5
                    * (self.T - t)
                    * bat.c_priority
                )
                objective += (
                    self.variables['d'][i][t]
                    * self.min_import_price
                    * 5e-5
                    * (self.T - t)
                    * bat.c_priority
                )

        self.problem += objective

    def _add_energy_balance_constraints(self):
        """Add constraints related to the energy balance to the model."""
        for t in self.time_steps:
            battery_net_discharge = 0
            for i, bat in enumerate(self.batteries):
                battery_net_discharge += -self.variables['c'][i][t] + self.variables['d'][i][t]

            # grid import
            e_grid_imp = self.variables['n'][t]
            if self.grid.p_max_imp is not None:
                if self.is_grid_demand_rate_active:
                    e_grid_imp = self.variables['n'][t] + self.variables['e_imp_lim_exc'][t]
                else:
                    e_grid_imp = self.variables['n'][t] + self.variables['e_imp_lim_exc'][t]

            # grid export
            e_grid_exp = self.variables['e'][t]
            if self.grid.p_max_exp is not None:
                e_grid_exp = self.variables['e'][t] + self.variables['e_exp_lim_exc'][t]

            # Managed loads sit on the demand side beside gt. They are not in gt:
            # the point is that the solver chooses when they draw.
            managed_draw = pulp.lpSum(
                self.variables['ml'][i][t] for i in range(len(self.managed_loads))
            )

            self.problem += (
                battery_net_discharge + self.time_series.ft[t] + e_grid_imp
                == e_grid_exp + self.time_series.gt[t] + managed_draw
            )

        # Grid flow direction constraints — per-slot tight M
        # M_e_t = max possible export in slot t = PV production + max battery discharge
        # M_n_t = max possible import in slot t = load demand + max battery charge
        # These are provably valid upper bounds and significantly tighter than the
        # global M, which closes the LP relaxation gap and reduces B&B tree size.
        _total_d_max = sum(b.d_max for b in self.batteries)
        _total_c_max = sum(b.c_max for b in self.batteries)
        for t in self.time_steps:
            _dt_h = self.time_series.dt[t] / 3600.0
            _m_exp_t = self.time_series.ft[t] + _total_d_max * _dt_h
            _m_imp_t = (
                self.time_series.gt[t] + _total_c_max * _dt_h
                + sum(load.max_power_w for load in self.managed_loads) * _dt_h
            )
            self.problem += self.variables['e'][t] <= _m_exp_t * self.variables['y'][t]
            self.problem += self.variables['n'][t] <= _m_imp_t * (1 - self.variables['y'][t])

        # Limit regular grid import power
        if self.grid.p_max_imp is not None:
            if self.is_grid_demand_rate_active:
                for t in self.time_steps:
                    self.problem += (
                        self.variables['n'][t]
                        <= self.grid.p_max_imp * self.time_series.dt[t] / 3600
                    )
                    self.problem += (
                        self.grid.p_max_imp * self.time_series.dt[t] / 3600
                        - self.variables['n'][t]
                        <= self.M * self.variables['z_imp_lim'][t]
                    )
                    self.problem += (
                        self.variables['e_imp_lim_exc'][t]
                        <= self.M * (1 - self.variables['z_imp_lim'][t])
                    )
            else:
                for t in self.time_steps:
                    self.problem += (
                        self.variables['n'][t]
                        <= self.grid.p_max_imp * self.time_series.dt[t] / 3600
                    )
                    self.problem += (
                        self.grid.p_max_imp * self.time_series.dt[t] / 3600
                        - self.variables['n'][t]
                        <= self.M * self.variables['z_imp_lim'][t]
                    )
                    self.problem += (
                        self.variables['e_imp_lim_exc'][t]
                        <= self.M * (1 - self.variables['z_imp_lim'][t])
                    )

        # Limit regular grid export power
        if self.grid.p_max_exp is not None:
            for t in self.time_steps:
                self.problem += (
                    self.variables['e'][t]
                    <= self.grid.p_max_exp * self.time_series.dt[t] / 3600
                )
                self.problem += (
                    self.grid.p_max_exp * self.time_series.dt[t] / 3600
                    - self.variables['e'][t]
                    <= self.M * self.variables['z_exp_lim'][t]
                )
                self.problem += (
                    self.variables['e_exp_lim_exc'][t]
                    <= self.M * (1 - self.variables['z_exp_lim'][t])
                )

        # Demand rate: track maximum import power
        if self.is_grid_demand_rate_active:
            for t in self.time_steps:
                self.problem += (
                    self.variables['e_imp_lim_exc'][t]
                    <= self.variables['p_max_imp_exc'] * self.time_series.dt[t] / 3600
                )

    def _add_battery_constraints(self):
        """Add constraints related to battery behavior to the model."""
        for i, bat in enumerate(self.batteries):
            # SOC limit penalties (handle out-of-range initial SOC)
            for t in range(0, self.T):
                self.problem += (
                    self.variables['s_max_pen'][i][t]
                    >= self.variables['s'][i][t] - bat.s_max
                )
                self.problem += (
                    self.variables['s_min_pen'][i][t]
                    >= bat.s_min - self.variables['s'][i][t]
                )

            # Battery dynamics
            if len(self.time_steps) > 0:
                self.problem += (
                    self.variables['s'][i][0]
                    == bat.s_initial
                    + self.eta_c * self.variables['c'][i][0]
                    - (1 / self.eta_d) * self.variables['d'][i][0]
                )
            for t in range(1, self.T):
                self.problem += (
                    self.variables['s'][i][t]
                    == self.variables['s'][i][t - 1]
                    + self.eta_c * self.variables['c'][i][t]
                    - (1 / self.eta_d) * self.variables['d'][i][t]
                )

            if bat.charge_curve is not None:
                self._add_charge_curve_constraints(i, bat)

            # SOC goal constraints (for t > 0)
            if bat.s_goal is not None:
                for t in range(1, self.T):
                    if bat.s_goal[t] > 0:
                        self.problem += (
                            self.variables['s'][i][t] + self.variables['s_goal_pen'][i][t]
                            >= bat.s_goal[t]
                        )

            # Minimum battery charge demand
            if bat.p_demand is not None:
                for t in self.time_steps:
                    if bat.p_demand[t] > 0:
                        p_demand = min(bat.c_max * self.time_series.dt[t] / 3600., bat.p_demand[t])
                        # two alternative constraints, only one is active:
                        self.problem += (
                            self.variables['c'][i][t] + self.variables['p_demand_pen'][i][t]
                            + self.M * self.variables['z_p_demand'][i][t]
                            >= p_demand
                        )
                        self.problem += (
                            self.variables['c'][i][t] + self.variables['p_demand_pen'][i][t]
                            + self.M * (1 - self.variables['z_p_demand'][i][t])
                            - (self.batteries[i].s_max - self.variables['s'][i][t])
                            >= 0.
                        )
                    elif bat.c_min > 0:
                        self.problem += (
                            self.variables['c'][i][t]
                            >= bat.c_min
                            * self.time_series.dt[t]
                            / 3600.
                            * self.variables['z_c'][i][t]
                        )
                        self.problem += (
                            self.variables['c'][i][t]
                            <= self.M * self.variables['z_c'][i][t]
                        )

            elif bat.c_min > 0:
                for t in self.time_steps:
                    self.problem += (
                        self.variables['c'][i][t]
                        >= bat.c_min
                        * self.time_series.dt[t]
                        / 3600.
                        * self.variables['z_c'][i][t]
                    )
                    self.problem += (
                        self.variables['c'][i][t]
                        <= self.M * self.variables['z_c'][i][t]
                    )

            # Control battery charging from grid — per-slot tight M
            if not bat.charge_from_grid:
                for t in self.time_steps:
                    _c_max_t = bat.c_max * self.time_series.dt[t] / 3600.0
                    self.problem += self.variables['c'][i][t] <= _c_max_t * self.variables['y'][t]

            # Control battery discharging to grid — per-slot tight M
            if not bat.discharge_to_grid:
                for t in self.time_steps:
                    _d_max_t = bat.d_max * self.time_series.dt[t] / 3600.0
                    self.problem += (
                        self.variables['d'][i][t]
                        <= _d_max_t * (1 - self.variables['y'][t])
                    )

            # Lock charging against discharging — per-slot tight M
            # Using c_max/d_max * dt per slot tightens the LP relaxation when z_cd
            # is fractional (matches the variable upper bounds exactly).
            for t in self.time_steps:
                _c_max_t = bat.c_max * self.time_series.dt[t] / 3600.0
                _d_max_t = bat.d_max * self.time_series.dt[t] / 3600.0
                self.problem += (
                    self.variables['d'][i][t]
                    <= _d_max_t * self.variables['z_cd'][i][t]
                )
                self.problem += (
                    self.variables['c'][i][t]
                    <= _c_max_t * (1 - self.variables['z_cd'][i][t])
                )

            # Emergency reserve constraint (EOS_connect extension)
            # Enforce s[i][T-1] >= s_reserve as a soft (penalized) constraint.
            # This means: s[i][T-1] + s_reserve_pen[i] >= s_reserve
            if bat.s_reserve > 0 and self.variables['s_reserve_pen'][i] is not None:
                self.problem += (
                    self.variables['s'][i][self.T - 1] + self.variables['s_reserve_pen'][i]
                    >= bat.s_reserve
                )

    @staticmethod
    def _binding_lines(lines, s_from, s_to, c_max):
        """The lines that dip below ``c_max`` somewhere between s_from and s_to."""
        kept = []
        for s_anchor, power, slope in lines:
            lowest = min(power + slope * (s_from - s_anchor),
                         power + slope * (s_to - s_anchor))
            if lowest < c_max:
                kept.append((float(s_anchor), float(power), float(slope)))
        return kept

    def _add_charge_curve_constraints(self, i, bat):
        """
        Hold each slot's charge under the battery's charging curve (EOS_connect extension).

        The curve is read at the slot's *average* state of charge - the mean of where
        it starts and where it ends - rather than at its start. At the steep end of a
        curve one 15-minute slot can move the battery several percent, and judging
        the whole slot by its first moment would plan more than the BMS lets in. The
        average is still linear in the variables: ``s[t]`` already contains this
        slot's charge, so the charge simply appears on both sides.

        A concave curve is the minimum of its lines and needs nothing else. A curve
        that bends upward somewhere - a hard drop followed by a long tail is the
        common shape - cannot be written that way. Slots listed in
        ``_curve_exact_slots`` then get a binary per concave piece that says which
        piece the state of charge is in; all others get the envelope, which is exact
        as long as they stay below ``envelope_exact_below``. :meth:`solve` moves a
        slot into the exact set when the plan takes it above that.
        """
        curve = bat.charge_curve
        cap = float(bat.s_capacity or 0.0)
        envelope = self._binding_lines(curve.envelope, 0.0, cap, bat.c_max)
        pieces = [
            (float(lo), float(hi), self._binding_lines(lines, lo, hi, bat.c_max))
            for lo, hi, lines in curve.pieces
        ]
        exact_slots = self._curve_exact_slots.get(i, set()) if len(pieces) > 1 else set()
        self.variables.setdefault('z_curve', {})[i] = {}

        for t in self.time_steps:
            dt_h = self.time_series.dt[t] / 3600.0
            s_start = bat.s_initial if t == 0 else self.variables['s'][i][t - 1]
            s_avg = 0.5 * (s_start + self.variables['s'][i][t])
            if t not in exact_slots:
                for s_anchor, power, slope in envelope:
                    self.problem += (
                        self.variables['c'][i][t]
                        <= dt_h * (power + slope * (s_avg - s_anchor))
                    )
                continue
            # Disaggregated ("convex hull") formulation: the slot's state of charge
            # and charge are split over the pieces, and only the chosen piece may
            # carry any. Its linear relaxation is exactly the envelope, so the
            # solver starts from the fast plan and only has to branch where the
            # envelope and the curve disagree - far tighter than big-M switches.
            z = [self._new_var(f"z_curve_{i}_{t}_{k}", cat='Binary')
                 for k in range(len(pieces))]
            s_k = [self._new_var(f"s_curve_{i}_{t}_{k}", lowBound=0)
                   for k in range(len(pieces))]
            c_k = [self._new_var(f"c_curve_{i}_{t}_{k}", lowBound=0)
                   for k in range(len(pieces))]
            self.variables['z_curve'][i][t] = z
            self.problem += pulp.lpSum(z) == 1
            self.problem += pulp.lpSum(s_k) == s_avg
            self.problem += pulp.lpSum(c_k) == self.variables['c'][i][t]
            c_cap = bat.c_max * dt_h
            for k, (s_from, s_to, lines) in enumerate(pieces):
                self.problem += s_k[k] >= s_from * z[k]
                self.problem += s_k[k] <= s_to * z[k]
                self.problem += c_k[k] <= c_cap * z[k]
                for s_anchor, power, slope in lines:
                    self.problem += (
                        c_k[k]
                        <= dt_h * (power * z[k] + slope * (s_k[k] - s_anchor * z[k]))
                    )

    @staticmethod
    def _curve_power_at(curve, s_wh):
        """Power the curve allows at *s_wh*: the minimum of the lines of its piece."""
        for s_from, s_to, lines in curve.pieces:
            if s_wh <= s_to + 1e-9:
                return min(power + slope * (s_wh - s_anchor)
                           for s_anchor, power, slope in lines)
        s_from, s_to, lines = curve.pieces[-1]
        return min(power + slope * (s_wh - s_anchor) for s_anchor, power, slope in lines)

    def _curve_slots_beyond_envelope(self):
        """
        Slots the last solve took above ``envelope_exact_below``, per battery.

        Above that point the envelope may promise more than the curve, below it the two
        agree. A slot up there that charges nothing obeys any curve, but it is taken
        all the same: picking only the charging ones lets the next solve simply move
        the charge into a neighbouring high slot, one round per move. Taking them all
        settles a plan in one or two rounds, and their few binaries cost little.
        """
        found = {}
        for i, bat in enumerate(self.batteries):
            curve = bat.charge_curve
            if curve is None or len(curve.pieces) <= 1:
                continue
            exact = self._curve_exact_slots.setdefault(i, set())
            s_vals = [pulp.value(var) or 0.0 for var in self.variables['s'][i]]
            c_vals = [pulp.value(var) or 0.0 for var in self.variables['c'][i]]
            new = set()
            for t in self.time_steps:
                if t in exact:
                    continue
                s_start = bat.s_initial if t == 0 else s_vals[t - 1]
                s_avg = 0.5 * (s_start + s_vals[t])
                if s_avg > curve.envelope_exact_below + 1e-6:
                    new.add(t)
            if new:
                found[i] = new
        return found

    def _add_managed_load_constraints(self):
        """
        Bound each managed load: how much, when, for how long at a time, how often.

        The demand is a ceiling rather than an equality on purpose. A window too narrow
        to hold it, or a price that never comes down, has to yield a short plan - the
        caller can see the shortfall and say so. An equality would make the whole model
        infeasible and take the household's schedule down with it.
        """
        for i, load in enumerate(self.managed_loads):
            run = self.variables['ml'][i]
            self.problem += pulp.lpSum(run) <= max(0.0, load.demand_wh)

            # Frost protection and the like: energy that must be placed whatever it
            # costs, so it is a floor the value term cannot argue with.
            if load.urgent_wh > 0:
                self.problem += pulp.lpSum(run) >= min(load.urgent_wh, load.demand_wh)

            feasible = load.feasible or [True] * self.T
            self._add_daily_cap_constraints(load, run, feasible)

            on = self.variables['ml_on'][i]
            if on is None:
                continue

            for t in self.time_steps:
                cap = load.max_power_w * self.time_series.dt[t] / 3600.0
                allowed = t < len(feasible) and feasible[t]
                if not allowed or cap <= 0:
                    self.problem += on[t] == 0
                    continue
                # The switch *is* the draw, not merely a permission to draw. Binding it
                # one way only - run <= cap * on - leaves the binary free: the solver
                # sets it everywhere at no cost, every minimum-run constraint is then
                # satisfied by a switch that is on while nothing runs, and the appliance
                # gets scheduled in isolated slots anyway.
                #
                # So a load with a minimum run, or with starts priced, is modelled as
                # on/off at its rated power, which is what a compressor does. The demand
                # ceiling may then be met a slot short rather than exactly; that is the
                # honest answer for something that cannot half-run.
                self.problem += run[t] == cap * on[t]

            self._add_commitment_constraints(load, on, feasible)
            self._add_switching_constraints(
                on, self.variables['ml_start'][i], int(load.min_runtime_slots),
                already_running=load.already_running or load.committed_on_slots > 0,
            )

    def _add_shared_budget_constraints(self):
        """
        Hold every managed load together under one site limit.

        A budget belongs to the installation, not to a load: it is what the wiring or
        the contactor can carry, and it is separate from the grid import limit because
        sun can cover a draw the grid never sees. Without it two loads both take the
        single cheapest hour and stack a sauna on top of a pool pump - a peak the house
        cannot draw, which the optimizer then sizes the battery for.

        Never below the largest single load. A budget smaller than one appliance is a
        misconfiguration, and reading it literally would silently stop that appliance
        ever running rather than sharing anything out.
        """
        budget_w = self.managed_load_budget_w
        if budget_w <= 0 or len(self.managed_loads) < 2:
            return
        floor_w = max(load.max_power_w for load in self.managed_loads)
        effective_w = max(budget_w, floor_w)

        for t in self.time_steps:
            allowance = effective_w * self.time_series.dt[t] / 3600.0
            self.problem += pulp.lpSum(
                self.variables['ml'][i][t] for i in range(len(self.managed_loads))
            ) <= allowance

    def _add_daily_cap_constraints(self, load, run, feasible):
        """
        Hold a load to its allowed hours per calendar day.

        The cap reached the fallback planner and not the solver, so under this backend
        it was silently inert - and the pool preset ships it set to twelve, so a user
        on defaults believed in a limit that was not there.

        Written against the energy rather than the on/off binary because that binary
        only exists for a load with a minimum run or a priced start; a load without
        either is modelled continuously and would slip the cap entirely. N slots at
        rated power is the same limit either way, and it is what the fallback planner
        means by it.

        Never below what the head of the horizon is already committed to: a load part
        way through a run it may not abandon, on a day whose allowance is spent, would
        otherwise make the model infeasible and take the household's schedule with it.
        This is a comfort limit, not one worth failing the whole solve over.
        """
        cap = int(load.max_slots_per_day or 0)
        days = load.day_index or []
        if cap <= 0 or not days:
            return

        held = min(int(load.committed_on_slots or 0), self.T)
        grouped = {}
        for t in self.time_steps:
            if t >= len(days):
                continue
            day = days[t]
            if day is None or day < 0:
                continue
            grouped.setdefault(day, []).append(t)

        for day, slots in grouped.items():
            usable = [t for t in slots if t < len(feasible) and feasible[t]]
            if not usable:
                continue
            allowed = max(cap, sum(1 for t in slots if t < held))
            if len(usable) <= allowed:
                continue
            # Slots are not all the same length once the horizon changes resolution,
            # so the allowance is the dearest `allowed` of them, not a slot count.
            budget = sum(sorted(
                (load.max_power_w * self.time_series.dt[t] / 3600.0 for t in usable),
                reverse=True,
            )[:allowed])
            self.problem += pulp.lpSum(run[t] for t in usable) <= budget

    def _add_commitment_constraints(self, load, on, feasible):
        """
        Hold the head of the horizon to what the appliance is already doing.

        Bounded rather than absolute, on both sides. A run is only held through slots
        the load is actually allowed to use - if the window closes or the weather turns
        while it is running, the rule that stopped it wins - and only as far as the
        demand reaches, so committing can never ask for energy the load does not want.
        Either would otherwise make the model infeasible and take the household's
        schedule down with it.
        """
        held_on = min(int(load.committed_on_slots), self.T)
        if held_on:
            budget = max(0.0, load.demand_wh)
            for t in range(held_on):
                cap = load.max_power_w * self.time_series.dt[t] / 3600.0
                allowed = t < len(feasible) and feasible[t]
                if not allowed or cap <= 0 or budget < cap:
                    break
                self.problem += on[t] == 1
                budget -= cap

        for t in range(min(int(load.committed_off_slots), self.T)):
            self.problem += on[t] == 0

    def _add_switching_constraints(self, on, starts, span, already_running=False):
        """
        How long a run lasts, how long a rest lasts, and what a start is worth avoiding.

        The rising edge is `on[t] - on[t-1]`. Slot 0 has nothing before it, so a run
        beginning there normally counts as a start - unless the appliance was *already*
        running, in which case slot 0 continues a run somebody else began and charging
        it as a start would demand a fresh minimum run the demand may not cover. A
        *stop* in slot 0 is never counted: what came before the window is only known
        when the caller says so.
        """
        def edge(t):
            if t == 0:
                return 0 if already_running else on[0]
            return on[t] - on[t - 1]

        for t in self.time_steps:
            started = edge(t)
            for step in range(1, span):
                if t + step < self.T:
                    self.problem += on[t + step] >= started

        # A stop obliges the same rest as a run, so the solver cannot drop a single slot
        # wherever the tariff ticks up and leave the compressor cycling for nothing.
        for t in range(1, self.T):
            stopped = on[t - 1] - on[t]
            for step in range(1, span):
                if t + step < self.T:
                    self.problem += on[t + step] <= 1 - stopped

        if starts is None:
            return
        # One inequality is enough: the objective pays for starts, so the solver drives
        # them to zero wherever it can and never needs the other direction pinned.
        for t in self.time_steps:
            self.problem += starts[t] >= edge(t)

    def _refine_charge_curve(self, solver, first_solve_s):
        """
        Re-solve with the exact curve wherever the envelope is not exact (see below).

        A charging curve that is not concave is first planned along its envelope,
        which is linear and fast. Wherever that plan takes the state of charge above
        the point where the envelope stops being exact, those slots get the exact
        formulation and the problem is solved again; the remaining slots stay below
        that point, where envelope and curve agree. Usually one or two rounds.

        The managed loads' switches (a heat pump's runtime blocks, the dishwasher's
        start) are held where the first solve put them. Those binaries are what makes
        a solve slow, and where the battery tops up near full hardly moves them: with
        them held, a round costs about what a plan without managed loads does, rather
        than a second full branch and bound. Their energy stays free within the held
        blocks, and so does everything the battery and the grid do.

        The rounds share a time budget. A round that runs out of it keeps the best
        plan the solver found by then - that one obeys the curve too. If it found
        none, the plan of the round before stands, and the log names the slots in
        which it may count on more than the curve allows.
        """
        budget = max(self.CURVE_REFINE_BUDGET_MIN_S,
                     self.CURVE_REFINE_BUDGET_FACTOR * first_solve_s)
        if self.settings.time_limit:
            budget = min(budget, float(self.settings.time_limit))
        deadline = time.monotonic() + budget
        own_limit = solver.timeLimit
        switches = {
            var.name: round(pulp.value(var) or 0.0)
            for var in self._managed_load_switches()
        }
        try:
            for _ in range(self.CURVE_REFINE_ROUNDS):
                if pulp.LpStatus[self.problem.status] != 'Optimal':
                    return
                beyond = self._curve_slots_beyond_envelope()
                if not beyond:
                    return
                remaining = deadline - time.monotonic()
                if remaining < 0.5:
                    self._warn_curve_not_refined(beyond, budget)
                    return
                previous = (
                    self.problem,
                    {key: dict(val) if isinstance(val, dict) else val
                     for key, val in self.variables.items()},
                    {i: set(slots) for i, slots in self._curve_exact_slots.items()},
                )
                for i, slots in beyond.items():
                    self._curve_exact_slots[i] |= slots
                self.create_model()
                for var in self._managed_load_switches():
                    if var.name in switches:
                        var.lowBound = var.upBound = switches[var.name]
                solver.timeLimit = remaining if own_limit is None else min(own_limit, remaining)
                self.problem.solve(solver)
                if pulp.LpStatus[self.problem.status] != 'Optimal':
                    self.problem, self.variables, self._curve_exact_slots = previous
                    self._warn_curve_not_refined(beyond, budget)
                    return
        finally:
            solver.timeLimit = own_limit

    def _managed_load_switches(self):
        """The managed loads' on/off and start binaries of the current model."""
        for key in ('ml_on', 'ml_start'):
            for per_load in (self.variables.get(key) or {}).values():
                yield from per_load or ()

    @staticmethod
    def _warn_curve_not_refined(beyond, budget):
        slots = sorted(t for found in beyond.values() for t in found)
        logger.warning(
            "[OPT-LocalEVopt] Charging curve: no exact plan within %.0f s - keeping "
            "the envelope plan, which may count on charging faster than the curve "
            "allows in %d slot(s) near full (first: %s)",
            budget, len(slots), slots[:5],
        )

    # Upper bound on re-solves for a non-concave charging curve (see solve()).
    CURVE_REFINE_ROUNDS = 6
    # Wall time the re-solves may take together: this many times the first solve, and
    # at least the floor. Branch and bound on the curve's binaries is usually quick,
    # but next to other binaries (a heat pump's runtime blocks, say) it can now and
    # then run long - and a plan that is a little hopeful near full is better than
    # one that arrives a minute late.
    CURVE_REFINE_BUDGET_FACTOR = 4.0
    CURVE_REFINE_BUDGET_MIN_S = 5.0

    def solve(self) -> Dict:
        """
        Creates the MILP model if none exists and solves the optimization problem.
        Returns a dictionary with the optimization results.
        """
        if self.problem is None:
            self.create_model()

        solver = _resolve_cbc_solver(
            msg=0,
            num_threads=self.settings.num_threads,
            time_limit=self.settings.time_limit,
            gapRel=self.settings.gapRel,
        )

        with TemporaryDirectory() as tmpdir:
            solver.tmpDir = tmpdir
            started = time.monotonic()
            self.problem.solve(solver)
            self._refine_charge_curve(solver, time.monotonic() - started)

        status = pulp.LpStatus[self.problem.status]

        e_grid_import = [pulp.value(var) or 0.0 for var in self.variables['n']]
        e_grid_export = [pulp.value(var) or 0.0 for var in self.variables['e']]

        # if demand rate is active, add the excess import back
        if self.is_grid_demand_rate_active:
            for t in self.time_steps:
                e_grid_import[t] += pulp.value(self.variables['e_imp_lim_exc'][t]) or 0.0

        # Limit violations
        grid_imp_limit_violated = False
        e_grid_imp_overshoot = []
        if self.grid.p_max_imp is not None:
            exc_vals = [pulp.value(var) or 0.0 for var in self.variables['e_imp_lim_exc']]
            grid_imp_limit_violated = max(exc_vals) > 0
            e_grid_imp_overshoot = exc_vals

        grid_exp_limit_hit = False
        e_grid_exp_overshoot = []
        if self.grid.p_max_exp is not None:
            exc_vals = [pulp.value(var) or 0.0 for var in self.variables['e_exp_lim_exc']]
            grid_exp_limit_hit = max(exc_vals) > 0
            e_grid_exp_overshoot = exc_vals

        if status == 'Optimal':
            result = {
                'status': status,
                'objective_value': self.get_clean_objective_value(),
                'limit_violations': {
                    'grid_import_limit_exceeded': grid_imp_limit_violated,
                    'grid_export_limit_hit': grid_exp_limit_hit,
                },
                'batteries': [],
                'grid_import': e_grid_import,
                'grid_export': e_grid_export,
                'flow_direction': [],
                'grid_import_overshoot': e_grid_imp_overshoot,
                'grid_export_overshoot': e_grid_exp_overshoot,
                'managed_loads': [
                    {
                        'id': load.id,
                        'energy': [
                            pulp.value(var) or 0.0 for var in self.variables['ml'][i]
                        ],
                    }
                    for i, load in enumerate(self.managed_loads)
                ],
            }
            for i, bat in enumerate(self.batteries):
                result['batteries'].append({
                    'charging_power': [pulp.value(var) or 0.0 for var in self.variables['c'][i]],
                    'discharging_power': [pulp.value(var) or 0.0 for var in self.variables['d'][i]],
                    'state_of_charge': [pulp.value(var) or 0.0 for var in self.variables['s'][i]],
                })
            for y_var in self.variables['y']:
                if y_var is not None:
                    result['flow_direction'].append(int(pulp.value(y_var) or 0))
                else:
                    result['flow_direction'].append(0)
            return result
        else:
            return {
                'status': status,
                'objective_value': None,
                'limit_violations': {
                    'grid_import_limit_exceeded': False,
                    'grid_export_limit_hit': False,
                },
                'batteries': [],
                'grid_import': [],
                'grid_export': [],
                'flow_direction': [],
                'grid_import_overshoot': [],
                'grid_export_overshoot': [],
            }

    def get_clean_objective_value(self):
        """Recalculate the objective value without penalties and strategy incentives.

        The battery term is measured from ``bat.s_initial``, not from ``s[i][0]``.
        Per the dynamics in _add_battery_constraints, ``s[i][0]`` is the SOC
        *after* the first slot has charged or discharged::

            s[i][0] == bat.s_initial + eta_c*c[i][0] - (1/eta_d)*d[i][0]

        so differencing against it drops the first slot from the reported
        delta, understating or overstating the battery contribution by
        ``(s[i][0] - bat.s_initial) * p_a``.  With a 15-minute slot and a
        10 kW charge limit that is up to ~2.3 kWh of SOC.

        Harmless while this value is only reported, but it is also what the
        managed-load counterfactual differences: two solves of the same horizon,
        one with the loads and one without.  The error cancels only if both
        solves happen to make the same first-slot decision, and placing a load
        is exactly what changes it -- so the residual lands straight on the
        measured ct/kWh.

        Note the solved objective uses the level ``s[T-1] * p_a`` rather than a
        delta.  That is equivalent for optimisation, since s_initial is a
        constant and the two differ only by a constant, but the delta is the
        right form for reporting what the plan actually cost.
        """
        clean_objective = 0
        for t in self.time_steps:
            if self.grid.p_max_imp is not None:
                clean_objective -= (
                    (pulp.value(self.variables['n'][t]) or 0.0)
                    + (pulp.value(self.variables['e_imp_lim_exc'][t]) or 0.0)
                ) * self.time_series.p_N[t]
            else:
                clean_objective -= (
                    (pulp.value(self.variables['n'][t]) or 0.0)
                    * self.time_series.p_N[t]
                )
        for t in self.time_steps:
            clean_objective += (pulp.value(self.variables['e'][t]) or 0.0) * self.time_series.p_E[t]
        for i, bat in enumerate(self.batteries):
            clean_objective += (
                (pulp.value(self.variables['s'][i][self.T - 1]) or 0.0)
                - bat.s_initial
            ) * bat.p_a
        if self.is_grid_demand_rate_active:
            clean_objective += -self.grid.prc_p_exc_imp * (
                pulp.value(self.variables['p_max_imp_exc']) or 0.0
            )
        return clean_objective
