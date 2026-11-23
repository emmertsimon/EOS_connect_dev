# EOS connect develop + Heizung (PR 368)

Test-Add-on: aktueller `ohAnd/EOS_connect` develop (0.3.42.400, inkl. #380 Ladekurve pro Slot) mit PR #368
(`feature/space-heating-managed-loads`, inkl. ohAnds Commits vom 14.11.) und dem
Ladekurven-PRs (`feature/charging-curve-points-v2` Stützpunkte, `feature/charging-curve-exact` genaue Planung) und dem Ladegrenzen-PR
(`feature/ha-inverter-charge-limit`, EOS schickt die max. Ladeleistung an HA) und der modulierenden
Wärmepumpe (`feature/modulating-heat-pump`, Teillast-COP) und der Leistungskurve
(`feature/heating-power-curve`: WP nur an/aus, Leistung pro Slot aus einer aus den Läufen
gelernten Formel über Außen- und Haustemperatur) und dem Soft-/Hard-Limit für Speicher
(`feature/dhw-soft-hard-limit`: unter `soft_limit_temp_c` eine Ladung im nächsten Fenster trotz
Preisgrenze, unter `frost_protection_temp_c` sofort) obendrauf.
Quellstand: Branch `test/develop400-heating-curve-limit` in diesem Repo.

- Wird auf dem HA-Host lokal gebaut (dauert einige Minuten).
- Eigene Datenbank, eigener Slug: das develop-Add-on bleibt unangetastet.
- `boot: manual`, Port außen 8082.
- Nie gleichzeitig mit dem develop- oder stabilen EOS-Add-on laufen lassen:
  beide würden den Wechselrichter und dieselben MQTT-Entitäten steuern.
