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
| `review_profiles`, `review` | Prüfprofile und Zusammenfassung ohne Transkripte; `labels_changed` zählt tatsächlich übernommene Negative |
| `speaker_routes` | konfigurierte Lautsprecherrouten |
| `extraction` | Freigabe und Verfügbarkeit der Spracherkennung |
| `deployment` | Stand des Firmware-Rollouts |
| `automatic_training` | Stand und Wartegrund der Nachtplanung |

```bash
python3 - <<'PY'
import json
from pathlib import Path
from urllib.request import HTTPRedirectHandler, Request, build_opener

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

token = Path("~/.config/wake-word-trainer/service.token").expanduser().read_text().strip()
request = Request("http://localhost:10701/v1/status", headers={"Authorization": f"Bearer {token}"})
with build_opener(NoRedirect).open(request, timeout=10) as response:
    print(json.dumps(json.load(response), indent=2, ensure_ascii=False))
PY
```

Das Beispiel liest die bestehende `token_file` des Dienstes (hier der Standardpfad).
Bei abweichender Einrichtung denselben Dateipfad verwenden. Der Header entsteht
im Prozess; der Token erscheint weder in Shellsubstitutionen noch Prozessargumenten
und wird bei einer HTTP-Umleitung nicht weitergereicht.

[Prüfschnittstelle zum Collector](home-assistant-review.md),
[Extraktion](extraction.md) und [Dienstkonfiguration](service.md) erklären
Authentifizierung, Datenrollen und erlaubte Parameter.
