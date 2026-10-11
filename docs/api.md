# HTTP API

The service started with `serve` listens on port 10701 by default. Every
`/v1` request needs `Authorization: Bearer <token>`.

| Request | |
| --- | --- |
| `GET /health` | `{"ok": true}`, no token |
| `GET /v1/status` | state of the current or last run, best result, GPU, resources, profiles, speaker routes, extraction, deployment, automatic training |
| `POST /v1/start` `{"profile": "<id>"}` | start a run with a configured profile (default `recommended`); 503 when the resource check refuses |
| `POST /v1/stop` | stop the run; finished rounds are kept |
| `GET /v1/report` | evaluation of the exported model (`report.json`) |
| `GET /v1/model/<slug>.json` / `.tflite` | the exported model for ESPHome |
| `POST /v1/speaker_test` `{"route": "<name>"}` | play one held-out recording through a configured loudspeaker route |
| `POST /v1/extract` | WAV body, header `X-Wakeword-Phrases` (JSON list): returns wake-word clips, see [extraction](extraction.md) |

Callers can only pick configured profiles and routes; no commands, paths or
settings can be sent.

## Status fields

| Field | |
| --- | --- |
| `state` | `starting`, `running`, `completed`, `stopped` or `failed` |
| `last_error` | reason of the last failure |
| `wake_word`, `slug` | project |
| `gpu` | detected GPU |
| `resources` | memory, load, VRAM and a warning while a run swaps |
| `profiles` | id → label |
| `speaker_routes` | configured loudspeaker routes |
| `extraction` | `enabled`, `available` |
| `deployment` | state of the firmware rollout |
| `automatic_training` | why the idle-night scheduler is waiting |

```bash
curl -s -H "Authorization: Bearer $(cat ~/.config/wake-word-trainer/service.token)" \
  http://localhost:10701/v1/status | python3 -m json.tool
```
