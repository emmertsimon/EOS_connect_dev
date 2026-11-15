"""
The charging curve planned inside the solver (eos.local_evopt_exact_charging_curve).

What the user is promised: with exact planning on, no slot counts on the battery taking
more than the curve allows at the slot's average state of charge - so a battery that
drops hard at 90 % is not planned at full power through the drop. The checks read the
solved numbers back rather than trusting the constraint.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

pulp = pytest.importorskip("pulp")

from src.interfaces.charging_curve import (  # noqa: E402
    concave_envelope,
    curve_power_at,
    full_range_points,
)
from src.interfaces.optimization_backends.local_evopt.optimizer import (  # noqa: E402
    BatteryConfig,
    ChargeCurveConfig,
    GridConfig,
    ManagedLoadConfig,
    Optimizer,
    OptimizationStrategy,
    OptimizerSettings,
    TimeSeriesData,
)
from src.interfaces.optimization_backends.optimization_backend_local_evopt import (  # noqa: E402
    LocalEVOptBackend,
)
from src.interfaces.optimization_interface import OptimizationInterface  # noqa: E402

CAP = 10000.0
C_MAX = 5000.0
T = 16
DT = 900  # 15-minute slots: one slot can move this battery up to ~12 %
ETA = 0.95
TAPER = [(0.0, 5000.0), (80.0, 5000.0), (100.0, 500.0)]  # concave


def _curve(points, cap=CAP, exact=True):
    """The curve as the backend hands it to the solver (see `_charge_curve_config`)."""
    backend = LocalEVOptBackend.__new__(LocalEVOptBackend)
    config = backend._charge_curve_config(sorted(points), cap)
    if exact:
        return config
    # The envelope alone - what the first solve of an exact plan starts from.
    return ChargeCurveConfig(
        pieces=[(0.0, cap, config.envelope)], envelope=config.envelope,
        envelope_exact_below=cap,
    )


def _solve(points=None, s_initial=6000.0, pv_wh=2000.0, sun_slots=T, exact=True):
    """
    Sun for *sun_slots* slots and none after, nothing for exporting it: whatever the
    battery does not take while the sun shines is lost, so the solver charges as much
    as it is allowed to, as early as it has to.
    """
    battery = BatteryConfig(
        s_min=500, s_max=CAP, s_initial=s_initial, c_max=C_MAX, d_max=C_MAX,
        p_a=0.0003, charge_from_grid=False, s_capacity=CAP,
        charge_curve=_curve(points, exact=exact) if points else None,
    )
    series = TimeSeriesData(
        dt=[DT] * T, gt=[100.0] * T,
        ft=[pv_wh] * sun_slots + [0.0] * (T - sun_slots),
        p_N=[0.0003] * T, p_E=[0.0] * T,
    )
    optimizer = Optimizer(
        strategy=OptimizationStrategy(), grid=GridConfig(p_max_imp=20000, p_max_exp=20000),
        batteries=[battery], time_series=series, eta_c=ETA, eta_d=ETA,
        optimizer_settings=OptimizerSettings(time_limit=60), M=60000,
    )
    result = optimizer.solve()
    assert result["status"] == "Optimal"
    bat = result["batteries"][0]
    return bat["charging_power"], bat["state_of_charge"]


def _avg_soc_percent(s_initial, soc, t):
    start = s_initial if t == 0 else soc[t - 1]
    return 50.0 * (start + soc[t]) / CAP


@pytest.mark.parametrize("s_initial", [6000.0, 8500.0, 9500.0])
def test_no_slot_charges_more_than_the_curve_allows(s_initial):
    charge, soc = _solve(TAPER, s_initial=s_initial)
    for t, energy in enumerate(charge):
        allowed_wh = curve_power_at(TAPER, _avg_soc_percent(s_initial, soc, t)) * DT / 3600
        assert energy <= allowed_wh + 1e-3, f"slot {t}: {energy:.1f} Wh > {allowed_wh:.1f} Wh"


def test_the_taper_slows_the_last_stretch_down():
    # Two sunny slots of 8 kW: enough to fill the battery from 80 % at full power.
    tapered, soc_tapered = _solve(TAPER, s_initial=8000.0, sun_slots=2)
    free, soc_free = _solve(None, s_initial=8000.0, sun_slots=2)
    # Without the curve the first slot charges at the full 5 kW and the battery is
    # full by the end of the second ...
    assert free[0] == pytest.approx(C_MAX * DT / 3600, abs=1.0)
    assert soc_free[1] == pytest.approx(CAP, abs=1.0)
    # ... with it, both slots are already into the taper,
    assert tapered[1] < tapered[0] < C_MAX * DT / 3600 + 1e-6
    # so less of the sun ends up in the battery.
    assert soc_tapered[1] < soc_free[1] - 300


def test_below_the_knee_the_curve_changes_nothing():
    """A curve flat at c_max where the battery is cannot cost anything."""
    charge_curve, soc_curve = _solve(TAPER, s_initial=2000.0, pv_wh=1000.0, sun_slots=4)
    charge_free, soc_free = _solve(None, s_initial=2000.0, pv_wh=1000.0, sun_slots=4)
    assert charge_curve[:4] == pytest.approx(charge_free[:4], abs=1e-3)
    assert soc_curve[-1] == pytest.approx(soc_free[-1], abs=1e-3)


CONVEX = [(0.0, 5000.0), (90.0, 5000.0), (95.0, 2000.0), (100.0, 1500.0)]


def test_the_envelope_alone_is_held_to_its_hull():
    """The first solve's view: a curve that bends upward is planned along its hull."""
    hull = concave_envelope(full_range_points(CONVEX))
    charge, soc = _solve(CONVEX, s_initial=8800.0, exact=False)
    for t, energy in enumerate(charge):
        allowed_wh = curve_power_at(hull, _avg_soc_percent(8800.0, soc, t)) * DT / 3600
        assert energy <= allowed_wh + 1e-3


@pytest.mark.parametrize("s_initial", [8800.0, 9300.0])
def test_exact_planning_holds_a_convex_curve_itself(s_initial):
    charge, soc = _solve(CONVEX, s_initial=s_initial, sun_slots=6, exact=True)
    for t, energy in enumerate(charge):
        allowed_wh = curve_power_at(CONVEX, _avg_soc_percent(s_initial, soc, t)) * DT / 3600
        assert energy <= allowed_wh + 1e-3, f"slot {t}: {energy:.1f} Wh > {allowed_wh:.1f} Wh"


def test_exact_planning_charges_less_than_the_envelope_promises():
    envelope_charge, envelope_soc = _solve(CONVEX, s_initial=9000.0, sun_slots=2, exact=False)
    exact_charge, exact_soc = _solve(CONVEX, s_initial=9000.0, sun_slots=2, exact=True)
    assert exact_soc[0] < envelope_soc[0] - 50


def test_exact_planning_out_of_time_keeps_the_envelope_plan(monkeypatch, caplog):
    """No time left for the exact round: the envelope plan stands, and the log says so."""
    envelope_charge, envelope_soc = _solve(CONVEX, s_initial=9000.0, sun_slots=2, exact=False)
    monkeypatch.setattr(Optimizer, "CURVE_REFINE_BUDGET_MIN_S", 0.0)
    monkeypatch.setattr(Optimizer, "CURVE_REFINE_BUDGET_FACTOR", 0.0)
    with caplog.at_level(logging.WARNING):
        charge, soc = _solve(CONVEX, s_initial=9000.0, sun_slots=2, exact=True)
    assert soc == pytest.approx(envelope_soc, abs=1e-3)
    assert charge == pytest.approx(envelope_charge, abs=1e-3)
    assert "no exact plan within" in caplog.text


def test_a_round_the_solver_cannot_finish_falls_back_to_the_round_before(monkeypatch, caplog):
    """
    The exact round gets what is left of the budget as its time limit. If the solver
    stops without any plan, the plan of the round before is the result - not an
    empty one - and the solver's own limit is back in place afterwards.
    """
    envelope_charge, envelope_soc = _solve(CONVEX, s_initial=9000.0, sun_slots=2, exact=False)
    real_solve = pulp.LpProblem.solve
    limits = []

    def solve_then_give_up(self, solver=None, **kwargs):
        limits.append(solver.timeLimit)
        result = real_solve(self, solver, **kwargs)
        if len(limits) > 1:
            self.status = pulp.LpStatusNotSolved
        return result

    monkeypatch.setattr(pulp.LpProblem, "solve", solve_then_give_up)
    with caplog.at_level(logging.WARNING):
        charge, soc = _solve(CONVEX, s_initial=9000.0, sun_slots=2, exact=True)
    assert len(limits) == 2
    assert limits[0] == 60
    assert 0 < limits[1] <= Optimizer.CURVE_REFINE_BUDGET_MIN_S
    assert soc == pytest.approx(envelope_soc, abs=1e-3)
    assert charge == pytest.approx(envelope_charge, abs=1e-3)
    assert "no exact plan within" in caplog.text


def test_the_exact_round_holds_the_managed_loads_where_the_first_solve_put_them(monkeypatch):
    """
    A heat pump's runtime blocks are what make a solve slow; the exact round keeps
    them as placed and only re-plans battery and grid, so it costs a fraction of the
    first solve. Checked on the model the exact round solves: every switch is pinned
    to the value the first solve gave it.
    """
    battery = BatteryConfig(
        s_min=500, s_max=CAP, s_initial=9000.0, c_max=C_MAX, d_max=C_MAX,
        p_a=0.0003, charge_from_grid=False, s_capacity=CAP,
        charge_curve=_curve(CONVEX, exact=True),
    )
    series = TimeSeriesData(
        dt=[DT] * T, gt=[100.0] * T, ft=[2000.0] * 2 + [0.0] * (T - 2),
        p_N=[0.0003] * T, p_E=[0.0] * T,
    )
    load = ManagedLoadConfig(
        id="wp", demand_wh=1500.0, max_power_w=1000.0, value_eur_per_wh=0.0005,
        min_runtime_slots=4, start_cost_eur=0.01,
    )
    optimizer = Optimizer(
        strategy=OptimizationStrategy(), grid=GridConfig(p_max_imp=20000, p_max_exp=20000),
        batteries=[battery], time_series=series, eta_c=ETA, eta_d=ETA,
        optimizer_settings=OptimizerSettings(time_limit=60), M=60000,
        managed_loads=[load],
    )
    real_solve = pulp.LpProblem.solve
    rounds = []

    def record(self, solver=None, **kwargs):
        switches = list(optimizer._managed_load_switches())
        rounds.append([(var.name, var.lowBound, var.upBound) for var in switches])
        result = real_solve(self, solver, **kwargs)
        rounds[-1] = (rounds[-1], {var.name: round(pulp.value(var)) for var in switches})
        return result

    monkeypatch.setattr(pulp.LpProblem, "solve", record)
    result = optimizer.solve()
    assert result["status"] == "Optimal"
    assert len(rounds) == 2, "the envelope plan should have needed one exact round"
    (first_bounds, first_values), (exact_bounds, exact_values) = rounds
    assert first_values and any(first_values.values()), "the load should run"
    assert all(lo in (0, None) and hi in (1, None) for _, lo, hi in first_bounds)
    assert {name: (lo, hi) for name, lo, hi in exact_bounds} == {
        name: (value, value) for name, value in first_values.items()
    }
    assert exact_values == first_values


def test_lines_that_can_never_bind_are_left_out():
    curve = _curve([(0.0, 9000.0), (100.0, 6000.0)])  # always above c_max
    battery = BatteryConfig(s_capacity=CAP, s_max=CAP, c_max=C_MAX, charge_curve=curve)
    assert Optimizer._binding_lines(curve.envelope, 0.0, CAP, C_MAX) == []
    optimizer = MagicMock()
    optimizer.problem = MagicMock()
    optimizer.variables = {"s": {0: [MagicMock()] * T}, "c": {0: [MagicMock()] * T}}
    optimizer.time_steps = range(T)
    optimizer.time_series = SimpleNamespace(dt=[DT] * T)
    optimizer._curve_exact_slots = {}
    optimizer._binding_lines = Optimizer._binding_lines
    Optimizer._add_charge_curve_constraints(optimizer, 0, battery)
    optimizer.problem.__iadd__.assert_not_called()


# -- through the backend ------------------------------------------------------------------


def _eos_request(initial_soc_pct=85, n_slots=192):
    return {
        "ems": {
            "pv_prognose_wh": [1500.0] * n_slots,
            "strompreis_euro_pro_wh": [0.0003] * n_slots,
            "einspeiseverguetung_euro_pro_wh": [0.0] * n_slots,
            "gesamtlast": [100.0] * n_slots,
            "preis_euro_pro_wh_akku": 0.0002,
        },
        "pv_akku": {
            "device_id": "battery1", "capacity_wh": 10000,
            "charging_efficiency": 0.95, "discharging_efficiency": 0.95,
            "max_charge_power_w": 5000, "initial_soc_percentage": initial_soc_pct,
            "min_soc_percentage": 5, "max_soc_percentage": 100,
        },
    }


def _curve_callable(points):
    def curve(soc):
        return curve_power_at(points, soc)
    curve.points = list(points)
    return curve


def _built(backend, **kwargs):
    built = []
    original = LocalEVOptBackend._build_optimizer

    def spy(self, *args, **kw):
        optimizer = original(self, *args, **kw)
        built.append((optimizer, kw))
        return optimizer

    with patch.object(LocalEVOptBackend, "_build_optimizer", spy):
        response, _ = backend.optimize(_eos_request(), **kwargs)
    assert "error" not in response
    return built


def test_exact_mode_plans_the_curve_in_one_model_in_wh():
    backend = LocalEVOptBackend(
        time_frame_base=900, time_zone=ZoneInfo("Europe/Berlin"), exact_charging_curve=True
    )
    built = _built(backend, charge_power_limit=_curve_callable(CONVEX))
    assert len(built) == 1, "no per-slot limit iteration in exact mode"
    optimizer, kwargs = built[0]
    assert kwargs.get("c_max_series") is None
    curve = optimizer.batteries[0].charge_curve
    assert curve.envelope_exact_below == pytest.approx(9000.0)  # 90 % of 10 kWh
    assert len(curve.pieces) == 2


def test_without_exact_mode_the_limits_are_iterated_as_before():
    backend = LocalEVOptBackend(time_frame_base=900, time_zone=ZoneInfo("Europe/Berlin"))
    built = _built(backend, charge_power_limit=_curve_callable(CONVEX))
    assert all(optimizer.batteries[0].charge_curve is None for optimizer, _ in built)
    assert built[0][1].get("c_max_series") is not None


def test_exact_mode_needs_points_and_otherwise_iterates():
    backend = LocalEVOptBackend(
        time_frame_base=900, time_zone=ZoneInfo("Europe/Berlin"), exact_charging_curve=True
    )
    built = _built(backend, charge_power_limit=lambda soc: curve_power_at(CONVEX, soc))
    assert built[0][1].get("c_max_series") is not None


def test_the_facade_hands_the_option_to_the_backend():
    config = {"source": "local_evopt", "local_evopt_exact_charging_curve": True}
    facade = OptimizationInterface(config, 900, ZoneInfo("Europe/Berlin"))
    assert facade.backend.exact_charging_curve is True


def test_a_curve_is_announced_once(caplog):
    backend = LocalEVOptBackend(
        time_frame_base=900, time_zone=ZoneInfo("Europe/Berlin"), exact_charging_curve=True
    )
    with caplog.at_level(logging.INFO):
        _built(backend, charge_power_limit=_curve_callable(CONVEX))
        _built(backend, charge_power_limit=_curve_callable(CONVEX))
    assert caplog.text.count("Charging curve planned exactly") == 1
