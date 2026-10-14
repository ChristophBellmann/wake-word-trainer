# Kommandozeile

```text
wake-word-trainer [-p PROJECT] [--downloads FOLDER] <command> [options]
```

`-p` wählt das Projekt mit `wakeword.yaml`; Standard ist das aktuelle
Verzeichnis. `--downloads` wählt einen gemeinsamen Downloadordner,
standardmäßig `<projekt>/downloads`. Optionen erscheinen auch mit `--help`.

| Befehl | Zweck |
| --- | --- |
| `init "<aktivierungswort>"` | Projekt anlegen; `--language`, `--collector-url`, `--token-file` |
| `download` | Korpora, Features und Stimmen; `--voice`, `--speech`, `--negative-set`, `--data` |
| `fetch` | freigegebene Collector-Aufnahmen und lokale Aufnahmeordner übernehmen |
| `data import <manifest>` / `data check` | [Datenrollen und Herkunft](data-import.md) importieren und prüfen; `--dry-run`, `--rehash` |
| `teacher import/train/score` | optionalen Log-Mel-Teacher übernehmen, trainieren oder bewerten |
| `experiment prepare/run` | eingefrorene Seed-Vergleiche ohne Rollout |
| `benchmark` | [GPU-Durchsatz](gpu-training-and-benchmarks.md) in getrennten Prozessen mit Aufwärmen und Wiederholungen |
| `audit` | lokale Quellen mit STT und optionalen Modellen prüfen, ohne Labels zu ändern |
| `review` | ungeprüfte Aufnahmen über HA abrufen; `--device`, `--since`, `--limit`, `--stt-model`, `--threads`, `--model`, `--teacher` |
| `generate` | Piper-Sprache erzeugen; `--force` neu erzeugen |
| `features` | augmentierte Microfrontend-Features bauen |
| `train` | Student trainieren; `--restart` ohne alten Checkpoint |
| `evaluate` | Erkennung, Fehlalarme und Schwellenkurve bewerten |
| `export` | ESPHome-Modell und Manifest schreiben |
| `run` | vollständiger Ablauf; `--rounds`, `--steps`, `--resume`, `--force`, `--prepare-only` |
| `compare` | mehrere Modelle am gleichen Audio vergleichen; wiederholbares `--model`, `--current` |
| `mine` | falsch ausgelöste negative Trainingsfenster sammeln |
| `status` | aktuellen oder letzten Lauf anzeigen |
| `deploy` | konfigurierten geprüften Rollout wiederholen; `--compare` vorher vergleichen |
| `serve --config <yaml>` | Trainer-Dienst für Home Assistant starten |

`wake-word-trainer --version` meldet die installierte Programmversion.
Privatberichte und Sprachdaten werden nicht in das öffentliche Git übernommen.
