# Schnellstart

Zuerst eine [passende CPU-, ROCm- oder CUDA-Umgebung](installation.md) prüfen.
Der [Wake Word Collector](https://github.com/ChristophBellmann/ha-wake-word-collector)
ist die HA-Quelle; alternativ lokale WAV-Ordner unter `recordings.folders`
eintragen.

## Token und Projekt

Den Export-Token aus den Collector-Optionen privat speichern:

```bash
mkdir -p ~/.config/wake-word-trainer
(umask 077; printf '%s\n' '<collector-token>' > ~/.config/wake-word-trainer/collector.token)
wake-word-trainer -p my_project init "Hey Nova" --language de \
  --collector-url http://homeassistant.local:8123 \
  --token-file ~/.config/wake-word-trainer/collector.token
wake-word-trainer -p my_project download --voice de_DE-thorsten-high --data
```

`--data` holt gemeinsame Korpora, Raumantworten und negative Features.
Speicherbedarf hängt von der Auswahl ab; Downloadordner können projektübergreifend
geteilt werden.

## Stimmen und Phonetik

```yaml
tts:
  voices: [voices/de_DE-thorsten-high.onnx]
  phrases: [hey nova, hei nova]
  negative_phrases: [hey service, nova, hey]
training:
  device: gpu
```

Mehrere Stimmen und Aussprachevarianten helfen, ersetzen aber keine echten
positiven Aufnahmen derselben Satelliten. Negative Phrasen sind ähnlich klingende
Sprache, auf die das Modell nicht reagieren soll.

## Vorbereiten, trainieren und vergleichen

```bash
wake-word-trainer -p my_project fetch
wake-word-trainer -p my_project data check
wake-word-trainer -p my_project run
```

Der Bericht zeigt Erkennung und Fehlalarme am getrennten Prüfbestand. Für
ungeprüfte Auslösungen zuerst [STT und Teacher über HA](home-assistant-review.md)
verwenden; bestätigte Labels werden danach beim nächsten `fetch` übernommen.

Das exportierte Manifest und TFLite-Modell liegen unter `export/`. ESPHome
kann das Manifest lokal oder über eine erreichbare URL beziehen:

```yaml
micro_wake_word:
  models:
    - model: my_word.json
```

Ein neues Modell erst nach [Bewertung und Regression](evaluation.md)
[ausrollen](deployment.md). Für weitere deutsche Sprache eignet sich
`download --speech mls_de` als zusätzliche negative Quelle.
