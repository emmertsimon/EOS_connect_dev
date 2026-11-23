"""
A charge a store's soft or hard limit calls for, placed by the solver.

The price limit is what keeps a hot water tank from being heated at 45 ct; the soft
limit is what keeps it from going cold on a day when nothing is cheaper. Below it the
charge has to land in its own slots whatever it costs - but in the cheapest of them,
and without ever making the model infeasible.
"""

import pytest

pulp = pytest.importorskip("pulp")

from src.interfaces.optimization_backends.local_evopt.optimizer import (  # noqa: E402
    GridConfig,
    ManagedLoadConfig,
    Optimizer,
    OptimizationStrategy,
    OptimizerSettings,
    TimeSeriesData,
)

T = 24
DT = 3600
DEAR = 0.00045
CAP = 0.00038


def _optimizer(loads, prices):
    series = TimeSeriesData(
        dt=[DT] * T, gt=[300.0] * T, ft=[0.0] * T, p_N=list(prices),
        p_E=[0.00008] * T,
    )
    return Optimizer(
        strategy=OptimizationStrategy(),
        grid=GridConfig(p_max_imp=20000, p_max_exp=20000),
        batteries=[], time_series=series,
        optimizer_settings=OptimizerSettings(time_limit=60), M=60000,
        managed_loads=loads,
    )


def _tank(**kwargs):
    base = {
        "id": "ww", "demand_wh": 6000.0, "value_eur_per_wh": CAP,
        "max_power_w": 2900.0, "min_runtime_slots": 1,
    }
    base.update(kwargs)
    return ManagedLoadConfig(**base)


def _placed(result):
    return next(e["energy"] for e in result["managed_loads"] if e["id"] == "ww")


def _window(start, end):
    return [start <= t < end for t in range(T)]


def test_without_a_must_charge_the_price_limit_keeps_the_tank_cold():
    """The situation on 23.11.: 42-52 ct all day against a 38 ct limit."""
    prices = [DEAR + 0.00001 * (t % 5) for t in range(T)]
    result = _optimizer([_tank(feasible=_window(10, 16))], prices).solve()
    assert sum(_placed(result)) == pytest.approx(0.0, abs=1.0)


def test_the_must_charge_is_placed_past_the_price_limit_in_the_cheapest_slot():
    prices = [DEAR] * T
    prices[13] = DEAR - 0.00003   # the cheapest hour inside the window
    prices[3] = 0.00001           # far cheaper, but outside it
    load = _tank(
        feasible=_window(10, 16), must_wh=2000.0, must_slots=_window(10, 16),
    )
    result = _optimizer([load], prices).solve()
    placed = _placed(result)
    assert result["status"] == "Optimal"
    assert placed[13] == pytest.approx(2000.0, rel=0.02)
    # Past the charge, the price limit still rules: nothing else is bought.
    assert sum(placed) == pytest.approx(2000.0, rel=0.02)


def test_a_must_charge_the_slots_cannot_hold_is_a_shortfall_not_an_infeasible_model():
    load = _tank(
        feasible=_window(15, 16), must_wh=5000.0, must_slots=_window(15, 16),
        min_runtime_slots=2,
    )
    result = _optimizer([load], [DEAR] * T).solve()
    assert result["status"] == "Optimal"
    assert sum(_placed(result)) <= 2900.0 + 1.0


def test_an_urgent_floor_against_a_closed_window_no_longer_breaks_the_solve():
    """
    25.10.: frost protection asked for a whole horizon's demand as a hard floor while
    the window was shut, and every solve came back infeasible.
    """
    load = _tank(
        demand_wh=7000.0, urgent_wh=7000.0, feasible=_window(10, 16),
        min_runtime_slots=2,
    )
    result = _optimizer([load], [DEAR] * T).solve()
    assert result["status"] == "Optimal"
    assert sum(_placed(result)) > 0
