# HTTP-API

Der Trainer-Dienst hört standardmäßig auf Port 10701. Alle `/v1`-Aufrufe
benötigen `Authorization: Bearer <token>`; der Token bleibt in einer privaten
Datei. HTTP-Antworten sind als privat und nicht zu speichern gekennzeichnet.

| Aufruf | Ergebnis |
| --- | --- |
| `GET /health` | `{"ok": true}`, ohne Token |
| `GET /v1/status` | aktueller oder letzter Zustand, Fortschritt, Ressourcen, Profile und Zusammenfassungen |
| `POST /v1/start` `{"profile": "<id>"}` | vorkonfiguriertes Profil starten; 409 bei aktivem Lauf, 503 bei Ressourcenmangel |
| `POST /v1/stop` | laufendes Profil stoppen, abgeschlossene Runden behalten |
| `GET /v1/report` | Bewertung des Exportmodells |
| `GET /v1/review` | privater Detailbericht der letzten Aufnahmeprüfung einschließlich Transkripten |
| `GET /v1/model/<slug>.json` / `.tflite` | Exportmodell und ESPHome-Manifest |
| `POST /v1/speaker_test` `{"route": "<name>"}` | Prüfaufnahme über eine konfigurierte Lautsprecherroute wiedergeben |
| `POST /v1/extract` | WAV und `X-Wakeword-Phrases` als JSON-Liste; erkannte Ausschnitte zurückgeben |

## Statusfelder

| Feld | Bedeutung |
| --- | --- |
| `state` | `starting`, `running`, `completed`, `stopped`, `failed` |
| `phase`, `step`, `round` | laufender Arbeitsschritt und Trainingsfortschritt |
| `last_error` | Fehlergrund |
| `wake_word`, `slug` | Projektkennung |
| `gpu`, `resources` | Gerät, RAM, Last und Warnungen |
| `profiles` | Profilkennung → sichtbarer Name |
| `review_profiles`, `review` | Prüfprofile und Zusammenfassung ohne Transkripte |
| `speaker_routes` | konfigurierte Lautsprecherrouten |
| `extraction` | Freigabe und Verfügbarkeit der Spracherkennung |
| `deployment` | Stand des Firmware-Rollouts |
| `automatic_training` | Stand und Wartegrund der Nachtplanung |

```bash
curl -s -H "Authorization: Bearer $(cat ~/.config/wake-word-trainer/service.token)" \
  http://localhost:10701/v1/status | python3 -m json.tool
```

[Prüfschnittstelle zum Collector](home-assistant-review.md),
[Extraktion](extraction.md) und [Dienstkonfiguration](service.md) erklären
Authentifizierung, Datenrollen und erlaubte Parameter.
