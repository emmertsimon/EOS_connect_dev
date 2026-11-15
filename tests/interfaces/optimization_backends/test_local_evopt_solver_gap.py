"""
How close to optimal the built-in optimizer stops, and what that is measured against.

CBC stops at a 1 % gap, relative to the whole objective. The terminal terms used to
credit the battery for every Wh it ends with and a heated store for every kelvin
above 0 C - constants that do not change the optimum but made the objective, and so
the slack the gap allows, far larger than anything the plan decides. With a house in
the plan that slack exceeded the cost of an evening's household load, and a solve
could stop at a plan that held a full battery through a dear evening and bought the
house's power instead. Both terms are now measured from where they start.

The charging-curve rounds of an exact plan are each a fresh branch and bound whose
plan is kept as found, so they solve to a much tighter gap of their own: at the main
solve's 1 % the loss could compound from round to round.
"""

import pytest

pulp = pytest.importorskip("pulp")

from src.interfaces.optimization_backends.local_evopt.optimizer import (  # noqa: E402
    BatteryConfig,
    GridConfig,
    ManagedLoadConfig,
    OptimizationStrategy,
    Optimizer,
    OptimizerSettings,
    ThermalStoreConfig,
    TimeSeriesData,
)

CT = 1e-5


def _house_store(t_initial=21.8):
    """A heated building: about 21 kWh per kelvin, heated by a heat pump."""
    return ThermalStoreConfig(
        capacity_wh_per_k=21248.0, loss_w_per_k=[51.0] * 24, ambient_c=[4.0] * 24,
        cop=[4.0] * 24, gain_wh=[0.0] * 24, t_initial=t_initial,
        t_min=21.0, t_target=21.5, t_max=22.5,
    )


def _optimizer(store=None, settings=None, s_initial=15000.0):
    slots = 24
    loads = []
    if store is not None:
        loads.append(ManagedLoadConfig(
            id="space_heating", demand_wh=3000.0, max_power_w=2800.0,
            value_eur_per_wh=38 * CT, min_runtime_slots=2, store=store,
        ))
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy="none", discharging_strategy="none"),
        grid=GridConfig(),
        batteries=[BatteryConfig(
            s_min=750.0, s_max=15000.0, s_initial=s_initial, c_max=6000.0, d_max=6000.0,
            p_a=38 * CT, charge_from_grid=True, discharge_to_grid=False,
        )],
        time_series=TimeSeriesData(
            dt=[3600] * slots, gt=[500.0] * slots, ft=[0.0] * slots,
            p_N=[44 * CT] * 8 + [40 * CT] * 16, p_E=[8 * CT] * slots,
        ),
        eta_c=0.96, eta_d=0.96, managed_loads=loads,
        optimizer_settings=settings,
    )


def test_the_house_is_credited_from_where_it_starts():
    """A store that ends where it started adds nothing to the objective."""
    optimizer = _optimizer(store=_house_store(t_initial=21.3))
    optimizer.create_model()
    store_term = optimizer._store_objective(0, optimizer.managed_loads[0])
    tm_end = optimizer.variables["tm_end"][0]
    tm_end.setInitialValue(21.3)
    for t, var in enumerate(optimizer.variables["tm_lo"][0]):
        var.setInitialValue(0.0)
        optimizer.variables["tm_hi"][0][t].setInitialValue(0.0)
    assert pulp.value(store_term) == pytest.approx(0.0, abs=1e-9)


def test_a_full_battery_left_untouched_adds_nothing_either():
    optimizer = _optimizer(s_initial=15000.0)
    optimizer.solve()
    # Only what the plan bought and spent is in the objective, so its size is what a
    # relative gap is a fraction of: with no strategy terms the solved objective is
    # the reported cost, battery measured from where it started - not 5.7 EUR more.
    assert pulp.value(optimizer.problem.objective) == pytest.approx(
        optimizer.get_clean_objective_value(), abs=1e-6
    )


def test_a_dear_evening_is_served_from_the_battery_with_a_house_in_the_plan():
    """A house store in the plan, a full battery, an evening dearer than the rest:
    the battery serves the evening."""
    optimizer = _optimizer(store=_house_store())
    result = optimizer.solve()
    assert result["status"] == "Optimal"
    discharge = result["batteries"][0]["discharging_power"]
    # The 44 ct hours come first, 500 Wh of house load each: the battery covers them.
    assert sum(discharge[:8]) >= 3500.0
    assert sum(result["grid_import"][:8]) < 600.0


def test_curve_rounds_solve_to_a_tight_gap_and_hand_the_solver_back(monkeypatch):
    """The main solve keeps its 1 %; each charging-curve round solves near exactly, so
    the rounds cannot compound the slack, and the solver's own gaps come back after."""
    from tests.interfaces.optimization_backends.test_local_evopt_exact_curve import (  # pylint: disable=import-outside-toplevel
        CONVEX,
        _solve,
    )

    real = pulp.LpProblem.solve
    seen = []
    solvers = []

    def record(self, solver=None, **kwargs):
        seen.append((solver.optionsDict.get("gapRel"), solver.optionsDict.get("gapAbs")))
        solvers.append(solver)
        return real(self, solver, **kwargs)

    monkeypatch.setattr(pulp.LpProblem, "solve", record)
    _solve(CONVEX, s_initial=9000.0, sun_slots=2, exact=True)
    assert len(seen) >= 2, "the curve should have needed a round"
    assert seen[0] == (OptimizerSettings().gapRel, OptimizerSettings().gapAbs)
    for gap_rel, gap_abs in seen[1:]:
        assert gap_rel == Optimizer.CURVE_REFINE_GAP_REL <= 0.001
        assert gap_abs == Optimizer.CURVE_REFINE_GAP_ABS
    assert solvers[0].optionsDict.get("gapRel") == OptimizerSettings().gapRel
    assert "gapAbs" not in solvers[0].optionsDict
