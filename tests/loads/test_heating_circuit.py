"""
A heating circuit the heat pump's own controller has switched off.

A weather-compensated controller shuts its circuit down in summer mode, and then no
release makes the pump heat. Live, that was a night of three releases, a "too cold"
guard and a house cooling by 0.4 K while the plan carried 7 kWh of heating that could
not happen. These tests pin what a ``circuit_sensor`` changes: nothing planned, nothing
offered, nothing released, a reason that names the cause - and that a gateway dropping
out is not mistaken for an off circuit.
"""

from datetime import timedelta

import pytest

from tests.loads.conftest import NOW

from src.loads.models.thermal import REASON_CIRCUIT_OFF
from src.loads.presets import TYPE_SPACE_HEATING

HOUSE = {
    "id": "house",
    "type": TYPE_SPACE_HEATING,
    "living_area_m2": 170,
    "construction": "massive",
    "emitter": "floor",
    "building_form": "detached",
    "insulation": "renovated",
    "temp_sensor": "sensor.rooms",
    "power_sensor": "sensor.hp_power",
    "target_temp": 21.5,
    "min_runtime_minutes": 90,
}
WITH_CIRCUIT = dict(HOUSE, circuit_sensor="sensor.hc2_state")


def _manager(make_manager, installation, entry, circuit=None, rooms=20.6, external=False):
    installation.sensors.update({"sensor.rooms": rooms, "sensor.hp_power": 0.0})
    if circuit is not None:
        installation.sensors["sensor.hc2_state"] = circuit
    installation.temperature = [10.0] * 48
    manager = make_manager([entry])
    manager.external_scheduler = external
    manager.run_cycle()
    return manager


def test_without_the_sensor_a_cold_house_is_released(make_manager, installation):
    manager = _manager(make_manager, installation, HOUSE)
    item = manager.instance("house")
    assert item.last_release["released"] is True
    assert item.last_demand.detail["circuit_off"] is False
    assert item.last_demand.detail["circuit_state"] is None


def test_a_circuit_reading_heating_changes_nothing(make_manager, installation):
    manager = _manager(make_manager, installation, WITH_CIRCUIT, circuit="heating")
    item = manager.instance("house")
    assert item.last_release["released"] is True
    assert item.last_demand.detail["circuit_off"] is False
    assert item.last_demand.detail["circuit_state"] == "heating"


def test_an_off_circuit_plans_and_releases_nothing(make_manager, installation):
    manager = _manager(make_manager, installation, WITH_CIRCUIT, circuit="off")
    item = manager.instance("house")

    assert item.last_release["released"] is False
    assert item.last_release["reason"] == REASON_CIRCUIT_OFF
    assert item.last_demand.reason == REASON_CIRCUIT_OFF
    assert sum(item.last_plan) == 0
    assert not any(item.last_demand.feasible)
    # What the house is missing stays on the card.
    assert item.last_demand.total_wh > 0
    assert manager.registry.snapshot() == []


def test_an_off_circuit_is_not_offered_to_the_optimizer(make_manager, installation):
    manager = _manager(
        make_manager, installation, WITH_CIRCUIT, circuit="off", external=True
    )
    item = manager.instance("house")
    assert item.last_demand.store is None
    assert manager.schedulable() == []
    assert item.last_release["reason"] == REASON_CIRCUIT_OFF


def test_switching_off_ends_a_minimum_runtime_hold(make_manager, installation):
    manager = _manager(make_manager, installation, WITH_CIRCUIT, circuit="heating")
    item = manager.instance("house")
    assert item.last_release["released"] is True

    installation.sensors["sensor.hc2_state"] = "off"
    manager.run_cycle()
    assert item.last_release["released"] is False
    assert item.last_release["reason"] == REASON_CIRCUIT_OFF


def test_coming_back_plans_again(make_manager, installation):
    manager = _manager(make_manager, installation, WITH_CIRCUIT, circuit="off")
    item = manager.instance("house")
    assert item.last_release["released"] is False

    installation.sensors["sensor.hc2_state"] = "heating"
    manager.run_cycle()
    assert item.last_release["released"] is True
    assert sum(item.last_plan) > 0


def test_a_gateway_dropout_keeps_the_last_state_for_a_while(
    make_manager, installation
):
    clock = {"now": NOW}
    manager = make_manager([WITH_CIRCUIT], clock=lambda: clock["now"])
    installation.sensors.update(
        {"sensor.rooms": 20.6, "sensor.hp_power": 0.0, "sensor.hc2_state": "off"}
    )
    installation.temperature = [10.0] * 48
    manager.run_cycle()
    item = manager.instance("house")
    assert item.last_demand.detail["circuit_off"] is True

    installation.sensors["sensor.hc2_state"] = "unavailable"
    clock["now"] = NOW + timedelta(minutes=10)
    manager.run_cycle()
    assert item.last_demand.detail["circuit_off"] is True, "a blip is not a new state"

    clock["now"] = NOW + timedelta(minutes=45)
    manager.run_cycle()
    assert item.last_demand.detail["circuit_off"] is False, "too old to rely on"


@pytest.mark.parametrize("reading", [None, "unknown", "unavailable", ""])
def test_an_unreadable_circuit_without_history_counts_as_on(
    make_manager, installation, reading
):
    manager = _manager(make_manager, installation, WITH_CIRCUIT, circuit=reading)
    item = manager.instance("house")
    assert item.last_demand.detail["circuit_off"] is False
    assert item.last_release["released"] is True


def test_the_off_states_are_configurable_and_case_blind(make_manager, installation):
    entry = dict(WITH_CIRCUIT, circuit_off_states="Off, Summer")
    manager = _manager(make_manager, installation, entry, circuit="SUMMER")
    assert manager.instance("house").last_demand.detail["circuit_off"] is True


def test_the_switch_off_is_logged_once(make_manager, installation, caplog):
    caplog.set_level("INFO", logger="__main__")
    manager = _manager(make_manager, installation, WITH_CIRCUIT, circuit="off")
    manager.run_cycle()
    manager.run_cycle()
    warnings = [r for r in caplog.records if "heating circuit reads" in r.getMessage()]
    assert len(warnings) == 1
