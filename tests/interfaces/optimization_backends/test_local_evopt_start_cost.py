"""
What a start costs: wear as a fixed amount, loss as energy bought where the run begins.

The setting exists for a heat pump that cycles because a pause saves a few cent. The
promise is twofold: a pause is still taken when it saves more than a start costs, and
the loss is bought like any other energy - at the start slot's tariff, from the battery
or from surplus PV - rather than at a guessed average.
"""

import pytest

pulp = pytest.importorskip("pulp")

from src.interfaces.optimization_backends.local_evopt.optimizer import (  # noqa: E402
    BatteryConfig,
    GridConfig,
    ManagedLoadConfig,
    Optimizer,
    OptimizationStrategy,
    OptimizerSettings,
    ThermalStoreConfig,
    TimeSeriesData,
)

T = 24
DT = 3600
CHEAP = 0.00030
FEED_IN = 0.00008
HOUSE_W = 300.0


def _optimizer(loads, prices, pv=None, battery=False, slots=T, dt=DT):
    batteries = [
        BatteryConfig(
            s_min=500, s_max=10000, s_initial=5000, c_max=5000, d_max=5000,
            p_a=prices[0], charge_from_grid=True, s_capacity=10000,
        )
    ] if battery else []
    series = TimeSeriesData(
        dt=[dt] * slots, gt=[HOUSE_W] * slots,
        ft=list(pv) if pv else [0.0] * slots,
        p_N=list(prices), p_E=[FEED_IN] * slots,
    )
    return Optimizer(
        strategy=OptimizationStrategy(), grid=GridConfig(p_max_imp=20000, p_max_exp=20000),
        batteries=batteries, time_series=series, eta_c=0.95, eta_d=0.95,
        optimizer_settings=OptimizerSettings(time_limit=60), M=60000,
        managed_loads=loads,
    )


def _split_or_bridge(**kwargs):
    """
    Twelve hours to place in fourteen allowed ones, the middle two a little dearer.

    Skipping them makes two runs; running through them makes one and costs
    2 kWh x 5 ct = 10 ct more.
    """
    prices = [CHEAP] * T
    prices[6] = prices[7] = CHEAP + 0.00005
    mask = [t < 14 for t in range(T)]
    base = {
        "id": "pump", "demand_wh": 12000.0, "value_eur_per_wh": 0.003,
        "max_power_w": 1000.0, "min_runtime_slots": 2, "feasible": mask,
    }
    base.update(kwargs)
    return _optimizer([ManagedLoadConfig(**base)], prices)


def _entry(result):
    return result["managed_loads"][0]


def _runs(energy):
    return sum(1 for t, value in enumerate(energy)
               if value > 1 and (t == 0 or energy[t - 1] <= 1))


def test_without_a_price_the_pause_is_taken():
    result = _split_or_bridge().solve()
    assert result["status"] == "Optimal"
    assert _runs(_entry(result)["energy"]) == 2


def test_wear_above_what_the_pause_saves_merges_the_runs():
    result = _split_or_bridge(start_wear_eur=0.15).solve()
    assert _runs(_entry(result)["energy"]) == 1
    assert sum(_entry(result)["energy"]) == pytest.approx(12000.0, rel=0.01)


def test_wear_below_what_the_pause_saves_still_lets_it_pause():
    """A price on starts, not a veto."""
    result = _split_or_bridge(start_wear_eur=0.05).solve()
    assert _runs(_entry(result)["energy"]) == 2


def test_the_loss_is_priced_like_energy_and_merges_the_runs_too():
    """Two kWh at 30 ct is 60 ct a start - far above the 10 ct the pause saves."""
    result = _split_or_bridge(start_loss_wh=2000.0).solve()
    assert _runs(_entry(result)["energy"]) == 1


def test_the_loss_is_drawn_in_the_slot_the_run_begins():
    opt = _split_or_bridge(start_loss_wh=700.0)
    result = opt.solve()
    energy = _entry(result)["energy"]
    starts = _entry(result)["starts"]
    first = next(t for t, value in enumerate(energy) if value > 1)
    assert starts[first] == 1
    assert sum(starts) == 1
    # No battery, no PV: the grid carries the house, the run and the loss.
    assert result["grid_import"][first] == pytest.approx(HOUSE_W + 1000.0 + 700.0, abs=1.0)
    assert result["grid_import"][first + 1] == pytest.approx(HOUSE_W + 1000.0, abs=1.0)


def test_a_negative_tariff_does_not_buy_phantom_starts():
    """
    Drawing the loss earns money below zero. A start bounded only from below would be
    claimed in every slot for that, without the appliance ever running there. Real
    starts may still pay below zero - that is what wear is for - but every start the
    plan counts is one the appliance makes.
    """
    prices = [-0.00020] * T
    mask = [t < 6 for t in range(T)]
    load = ManagedLoadConfig(
        id="pump", demand_wh=4000.0, value_eur_per_wh=0.003, max_power_w=1000.0,
        min_runtime_slots=2, feasible=mask, start_loss_wh=500.0,
    )
    result = _optimizer([load], prices).solve()
    entry = _entry(result)
    assert sum(entry["starts"]) == _runs(entry["energy"])
    assert sum(entry["starts"]) < 6


def test_a_start_already_under_way_draws_no_loss():
    prices = [CHEAP] * T
    load = ManagedLoadConfig(
        id="pump", demand_wh=3000.0, value_eur_per_wh=0.003, max_power_w=1000.0,
        min_runtime_slots=2, committed_on_slots=2, already_running=True,
        start_loss_wh=500.0,
    )
    result = _optimizer([load], prices).solve()
    entry = _entry(result)
    assert entry["energy"][0] > 1
    assert entry["starts"][0] == 0
    assert result["grid_import"][0] == pytest.approx(HOUSE_W + 1000.0, abs=1.0)


def test_wear_is_reported_with_the_plan_the_tie_breaker_is_not():
    """Same plan both times - one run - so the reported cost differs by the wear alone."""
    plain = _split_or_bridge(start_cost_eur=0.15)
    plain.solve()
    worn = _split_or_bridge(start_wear_eur=0.15)
    worn.solve()
    assert plain.get_clean_objective_value() - worn.get_clean_objective_value() == (
        pytest.approx(0.15, abs=1e-6)
    )


def test_pv_makes_the_loss_almost_free():
    """
    The same start in a sunny slot costs only the feed-in it gives up. With a surplus
    in the morning and a dear evening, wear and loss together do not move the run out
    of the sun.
    """
    prices = [CHEAP * 2] * T
    pv = [0.0] * T
    for t in range(9, 14):
        pv[t] = 3000.0
    mask = [t >= 8 for t in range(T)]
    load = ManagedLoadConfig(
        id="pump", demand_wh=4000.0, value_eur_per_wh=0.003, max_power_w=1000.0,
        min_runtime_slots=2, feasible=mask, start_loss_wh=1000.0, start_wear_eur=0.10,
    )
    result = _optimizer([load], prices, pv=pv).solve()
    entry = _entry(result)
    first = next(t for t, value in enumerate(entry["energy"]) if value > 1)
    assert 9 <= first <= 10
    assert result["grid_import"][first] == pytest.approx(0.0, abs=1.0)


def _house(**kwargs):
    """The measured house: 33.9 kWh/K, 101 W/K, quarter hours, a 1.6 kW pump."""
    slots = 96
    ambient = [1.0] * slots
    store = ThermalStoreConfig(
        capacity_wh_per_k=33900.0, loss_w_per_k=[101.0] * slots, ambient_c=ambient,
        cop=[3.8] * slots, gain_wh=[15.0] * slots,
        t_initial=21.3, t_min=21.0, t_max=22.5, t_target=21.5,
    )
    base = {"id": "house", "max_power_w": 1600.0, "min_runtime_slots": 6, "store": store}
    base.update(kwargs)
    return ManagedLoadConfig(**base)


def _evening_prices():
    """26.11. from 20:00: 55 ct falling to 50 ct by 22:30, 45.6 ct through the night."""
    prices = []
    for slot in range(96):
        hour = 20 + slot / 4
        if hour < 22:
            prices.append(0.00055 - 0.00001 * slot)
        elif hour < 24:
            prices.append(0.00050 - 0.00004 * (slot - 8) / 8 * 1.1)
        elif hour < 30:
            prices.append(0.000456)
        elif hour < 33:
            prices.append(0.00062)
        else:
            prices.append(0.00056)
    return prices


def _house_runs(result):
    return _runs(_entry(result)["energy"])


def test_the_house_runs_no_more_often_with_starts_priced():
    prices = _evening_prices()
    cheap_start = _optimizer([_house(start_cost_eur=0.13)], prices, slots=96, dt=900).solve()
    priced = _optimizer(
        [_house(start_wear_eur=0.15, start_loss_wh=500.0)], prices, slots=96, dt=900
    ).solve()
    assert cheap_start["status"] == priced["status"] == "Optimal"
    assert _house_runs(priced) <= _house_runs(cheap_start)


def test_a_priced_start_never_lets_the_house_go_cold():
    """It moves heat between hours. The comfort floor is not what it trades."""
    prices = _evening_prices()
    result = _optimizer(
        [_house(start_wear_eur=2.0, start_loss_wh=3000.0)], prices, slots=96, dt=900
    ).solve()
    temps = _entry(result)["temperature"]
    assert min(temps) >= 21.0 - 0.02
