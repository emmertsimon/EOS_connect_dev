# EOS connect develop + Heizung (PR 368)

Test-Add-on: aktueller `ohAnd/EOS_connect` develop (0.3.42.398) mit PR #368
(`feature/space-heating-managed-loads`, inkl. ohAnds Commits vom 14.11.) und dem
Ladekurven-PR (`feature/charging-curve-points`) und dem Ladegrenzen-PR
(`feature/ha-inverter-charge-limit`, EOS schickt die max. Ladeleistung an HA) obendrauf.
Quellstand: Branch `test/develop-heating-curve` in diesem Repo.

- Wird auf dem HA-Host lokal gebaut (dauert einige Minuten).
- Eigene Datenbank, eigener Slug: das develop-Add-on bleibt unangetastet.
- `boot: manual`, Port außen 8082.
- Nie gleichzeitig mit dem develop- oder stabilen EOS-Add-on laufen lassen:
  beide würden den Wechselrichter und dieselben MQTT-Entitäten steuern.
