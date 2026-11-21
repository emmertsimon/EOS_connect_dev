"""
A price on battery wear in the built-in optimizer.

The round trip alone lets the model cycle the battery for a few cents - charging at
35 ct, say, to replace 38-41 ct later on. A wear cost per kWh charged
has to stop that kind of trade while leaving the ones worth the wear - storing PV for
a dear evening, or a wide spread - alone.
"""

import pulp
import pytz
import pytest

from src.config_web.hot_reload import _LOCAL_EVOPT_FIELD_MAP
from src.interfaces.optimization_backends.local_evopt.optimizer import (
    BatteryConfig,
    GridConfig,
    OptimizationStrategy,
    Optimizer,
    TimeSeriesData,
)
from src.interfaces.optimization_backends.optimization_backend_local_evopt import (
    LocalEVOptBackend,
)

CT = 1e-5  # EUR per Wh in one ct/kWh


def _optimizer(prices_ct, pv_wh=None, wear_ct=0.0, load_wh=500.0, s_initial=1000.0):
    slots = len(prices_ct)
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy="none", discharging_strategy="none"),
        grid=GridConfig(),
        batteries=[BatteryConfig(
            s_min=500.0, s_max=15000.0, s_initial=s_initial,
            c_min=0.0, c_max=5000.0, d_max=5000.0,
            # Leftover charge valued at the cheapest price ahead, as cheapest_ahead does.
            p_a=min(prices_ct) * CT,
            charge_from_grid=True, discharge_to_grid=False,
            p_wear=wear_ct * CT,
        )],
        time_series=TimeSeriesData(
            dt=[3600] * slots,
            gt=[load_wh] * slots,
            ft=list(pv_wh or [0.0] * slots),
            p_N=[p * CT for p in prices_ct],
            p_E=[8 * CT] * slots,
        ),
        eta_c=0.96, eta_d=0.96,
    )


def _charge(optimizer):
    """Planned charge per slot in Wh."""
    return [pulp.value(var) or 0.0 for var in optimizer.variables["c"][0]]


def _grid_charge_wh(optimizer, prices_ct, pv_wh=None):
    """Energy charged in slots where there is no PV to charge from."""
    optimizer.solve()
    charge = _charge(optimizer)
    pv = pv_wh or [0.0] * len(prices_ct)
    return sum(c for c, p in zip(charge, pv) if p <= 0)


# Today's shape: the cheapest hour of two days at 35 ct, a night at 38-39 ct.
NARROW = [35.0, 35.0] + [38.5] * 10


def test_without_wear_the_narrow_spread_is_traded():
    optimizer = _optimizer(NARROW)
    assert _grid_charge_wh(optimizer, NARROW) > 1000.0


def test_wear_stops_trading_a_narrow_spread():
    optimizer = _optimizer(NARROW, wear_ct=4.0)
    assert _grid_charge_wh(optimizer, NARROW) == pytest.approx(0.0, abs=1.0)


def test_a_wide_spread_still_pays_for_the_wear():
    wide = [20.0, 20.0] + [45.0] * 10
    optimizer = _optimizer(wide, wear_ct=4.0)
    assert _grid_charge_wh(optimizer, wide) > 1000.0


def test_pv_for_a_dear_evening_is_still_stored():
    """Exporting earns 8 ct; keeping it for 40 ct is worth far more than 4 ct wear."""
    prices = [40.0] * 12
    pv = [3000.0] * 4 + [0.0] * 8
    optimizer = _optimizer(prices, pv_wh=pv, wear_ct=4.0)
    optimizer.solve()
    charged = sum(_charge(optimizer)[:4])
    assert charged > 5000.0


def test_wear_is_part_of_the_reported_cost():
    plain = _optimizer(NARROW, pv_wh=[3000.0] * 2 + [0.0] * 10)
    plain.solve()
    worn = _optimizer(NARROW, pv_wh=[3000.0] * 2 + [0.0] * 10, wear_ct=4.0)
    worn.solve()
    # Same PV stored either way, so the difference is the wear on it.
    assert worn.get_clean_objective_value() < plain.get_clean_objective_value()


@pytest.mark.parametrize("configured, expected", [
    (4.0, 4.0 * CT),
    (0.0, 0.0),
    (-3.0, 0.0),
    (None, 0.0),
    ("nonsense", 0.0),
    (float("nan"), 0.0),
])
def test_the_backend_turns_the_setting_into_eur_per_wh(configured, expected):
    backend = LocalEVOptBackend(
        time_frame_base=900, time_zone=pytz.timezone("Europe/Berlin"),
        battery_wear_ct_kwh=configured,
    )
    assert backend._wear_eur_per_wh() == pytest.approx(expected)


def test_the_backend_passes_it_into_the_battery():
    backend = LocalEVOptBackend(
        time_frame_base=3600, time_zone=pytz.timezone("Europe/Berlin"),
        battery_wear_ct_kwh=5.0,
    )
    request = {
        "batteries": [{"s_max": 15000, "s_min": 750, "s_initial": 5000,
                       "c_max": 6000, "d_max": 6000, "p_a": 0.0003}],
        "time_series": {"dt": [3600] * 4, "gt": [500.0] * 4, "ft": [0.0] * 4,
                        "p_N": [0.0003] * 4, "p_E": [0.00008] * 4},
    }
    optimizer = backend._build_optimizer(request, time_limit_s=10)
    assert optimizer.batteries[0].p_wear == pytest.approx(5.0 * CT)
    # Hot reload changes the attribute the next build reads.
    backend.battery_wear_ct_kwh = 0.0
    assert backend._build_optimizer(request, time_limit_s=10).batteries[0].p_wear == 0.0


def test_the_setting_is_hot_reloadable():
    assert _LOCAL_EVOPT_FIELD_MAP["eos.local_evopt_battery_wear_ct_kwh"] == (
        "battery_wear_ct_kwh", float
    )
