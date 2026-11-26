# EOS connect develop + Heizung (PR 368)

Test-Add-on: aktueller `ohAnd/EOS_connect` develop (0.3.42.400, inkl. #380 Ladekurve pro Slot) mit PR #368
(`feature/space-heating-managed-loads`, inkl. ohAnds Commits vom 14.11.) und dem
Ladekurven-PRs (`feature/charging-curve-points-v2` Stützpunkte, `feature/charging-curve-exact` genaue Planung) und dem Ladegrenzen-PR
(`feature/ha-inverter-charge-limit`, EOS schickt die max. Ladeleistung an HA) und der modulierenden
Wärmepumpe (`feature/modulating-heat-pump`, Teillast-COP) und der Leistungskurve
(`feature/heating-power-curve`: WP nur an/aus, Leistung pro Slot aus einer aus den Läufen
gelernten Formel über Außen- und Haustemperatur) und dem Soft-/Hard-Limit für Speicher
(`feature/dhw-soft-hard-limit`: unter `soft_limit_temp_c` eine Ladung im nächsten Fenster trotz
Preisgrenze, unter `frost_protection_temp_c` sofort; mit 0,3 K Hysterese `limit_hysteresis_k`
und Überbrückung eines bis 15 min ausgefallenen Temperatursensors) und dem Hauslernen
(`feature/learn-house-capacity`: das Haus lernt Speicherkapazität, Wärmeverlust, Sonnen- und
Eigenwärme aus seinen 12-h-Vorhersagen, übernommen nur wenn sie die letzten 2 Tage besser
vorhergesagt hätten) und dem Fix für Prognose-Ausfälle (`fix/pv-forecast-outage-fallback`:
bei nicht erreichbarem Prognosedienst nimmt jede PV-Fläche ihre eigene letzte Kurve statt der
Summe, keine Vervielfachung mehr; die Außentemperatur bleibt bei der letzten echten Prognose
statt 15 °C; zerschossene Prognosestunden zählen nicht in die PV-Autoskalierung) und den
Startkosten (`feature/heat-pump-start-cost`: `start_wear_ct` als fester Verschleiß pro Start,
`start_loss_kwh` als Anlaufverlust, den der Optimierer im Startslot zu dessen Preis, aus dem
Akku oder aus PV kauft; ersetzt den kleinen Gleichstand-Brecher, beide 0 = wie bisher) obendrauf.
Quellstand: Branch `test/develop400-heating-curve-limit` in diesem Repo.

- Wird auf dem HA-Host lokal gebaut (dauert einige Minuten).
- Eigene Datenbank, eigener Slug: das develop-Add-on bleibt unangetastet.
- `boot: manual`, Port außen 8082.
- Nie gleichzeitig mit dem develop- oder stabilen EOS-Add-on laufen lassen:
  beide würden den Wechselrichter und dieselben MQTT-Entitäten steuern.
