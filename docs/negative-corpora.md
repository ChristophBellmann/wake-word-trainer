# Negativkorpora aus dem Wakeword Lab

Die vorgefertigten Korpus-Negative werden bereits verwendet: `speech`,
`dinner_party` und `no_speech` gehören zu den Standardquellen des Students.
`dinner_party_eval` ist ausschließlich Ambient für Validation und Prüfung,
mit Sampling-Gewicht null. Zusätzliche WAV-Korpora kommen getrennt hinzu.
Diese Microfrontend-MMAPs gehören nicht in den Log-Mel-Teacher.

## Was tatsächlich vorhanden ist

Inventur vom 14.10.2026. Die Zahlen zählen MMAP-Einträge, keine individuellen
Sprecher oder unabhängigen Gespräche. Wiederholte Ausschnitte derselben
Aufnahme sind nicht unabhängige Beobachtungen.

| HF-Archiv | Trainings-Einträge im alten Bestand | Bisherige neue Läufe | Standardgewicht |
| --- | ---: | --- | ---: |
| `speech` | 350.535 | als Negative enthalten | 10 |
| `dinner_party` | 58.307 | als Negative enthalten | 10 |
| `no_speech` | 238.441 | als Negative enthalten | 5 |
| `dinner_party_eval` | 0; 4 Validation- und 10 Prüfaufnahmen | ausschließlich Ambient-Holdout | 0 |
| `speech_background` | 630.158 | nicht enthalten; alte 20-ms-Features | — |
| `dinner_party_background` | 29.153; zusätzlich Ambient-Holdouts | nicht enthalten; alte 20-ms-Features | — |
| `no_speech_background` | 228.518 | nicht enthalten; alte 20-ms-Features | — |

Die ersten vier entpackten Archive belegen hier ungefähr 25 GiB. Die drei
zusätzlichen belegen im alten Bestand weitere rund 17 GiB. Alle sieben
Archive lassen sich herunterladen; das bedeutet nicht, dass alle mit dem
aktuellen Frontend kompatibel sind.

Die [HF-Quellenbeschreibung](https://huggingface.co/datasets/kahrendt/microwakeword/blob/main/README.md)
nennt für die älteren Background-Archive 20-ms-Schritte und beschreibt
CHiME6/DipCo-Gespräche, LibriSpeech/VOiCES sowie FMA/FSD50K/WHAM. Der Trainer
und die ESPHome-Modelle benötigen 10-ms-Features. Im lokalen DipCo-Bestand
enthält die ältere Background-Fassung für dasselbe Audio halb so viele
Frames wie `dinner_party_eval`. Die Auswahl eines vorhandenen alten
`*_background`-Archivs wird deshalb vor dem Training mit einem konkreten
Fehler abgelehnt. Ein Float-Cast oder Verdoppeln der Frames regeneriert
keine passenden Microfrontend-Features. Dafür die Original-WAVs verwenden.

Unter dem Namen „Coffee Talk“ fand sich in den geprüften Lab-Skripten,
Manifesten und Downloadquellen kein eigener Datensatz. `dinner_party` ist
anhand seiner MMAP-Namen als CHiME-Gesprächsbestand identifiziert. Eine
Gleichsetzung mit einem weiteren namensähnlichen Korpus ist nicht belegt.

## Download und aktive Auswahl

```bash
wake-word-trainer -p project --downloads shared_downloads download \
  --negative-set speech --negative-set dinner_party \
  --negative-set no_speech --negative-set dinner_party_eval
```

```yaml
negatives:
  feature_sets: [speech, dinner_party, no_speech, dinner_party_eval]
training:
  weights:
    speech: 10
    dinner_party: 10
    no_speech: 5
```

Der Download legt Daten ab; `negatives.feature_sets` bestimmt die Auswahl.
Fehlende konfigurierte Trainingsquellen werden im Log ausdrücklich gemeldet.
`training_parameters.yaml` dokumentiert die tatsächlich verwendeten Pfade,
Gewichte und Datenrollen. Dort nachprüfen, statt aus vorhandenen ZIP-Dateien
oder einem alten YAML-Manifest auf die Nutzung zu schließen.

`dinner_party_eval` bleibt bei vorhandenem Download für die Bewertung erhalten,
auch wenn die drei Trainingsquellen aus `feature_sets` entfernt werden. Das
ermöglicht einen Vergleich auf unverändertem Ambient-Audio.

## Welche alten Hilfsmittel weitergeführt werden

| Altes Werkzeug | Heutiger Ablauf |
| --- | --- |
| `scripts/setup/setup_sync_datasets.py` | `download --negative-set`, Sprach-Presets, `data import`, `features`; vorhandene kompatible MMAPs direkt wiederverwenden |
| `scripts/setup/setup_fetch_notebook_backgrounds.py` | vorhandene AudioSet-/FMA-WAVs als `negative_background` importieren; eigene Features mit 10 ms erzeugen |
| `scripts/train_from_config.sh` | `wakeword.yaml` und `train`/`run`; tatsächlich erzeugte Backend-Konfiguration nachsehen |
| `docs/training_dataset.md` | Rollen weiterhin trennen: Korpus-Negative, WAV-Augmentation, Ambient-Holdout und RIR |

In den eingefrorenen Vergleichsläufen sind bereits 15.839 Sprach-WAVs aus
VoxPopuli Deutsch/Englisch und LibriSpeech sowie 41.235 AudioSet-/FMA-WAVs
übernommen. Ihre inhaltlich getrennten Training-/Validation-/Testseiten
bleiben erhalten. Raumimpulse und die Hintergrundstichprobe für
Audio-Augmentation erfüllen andere Rollen.

## Vergleich mit und ohne HF-Korpus-Negative

Die frühere [Ambient-Skalierung wurde korrigiert](evaluation.md#korrektur-der-mmap-skalierung-am-14102026).
Frühere FAPH-Zahlen und davon abhängige Parity-Freigaben müssen neu berechnet
werden. Eigene WAV-Trefferzahlen waren davon nicht betroffen.

Der kontrollierte Vergleich verwendet das offizielle Modellprofil, Batch 256,
GPU, optimierten TFRecord-Eingang, dieselben eingefrorenen Features, Seeds
`2026070716` und `2026070722`, identischen Frühstopp sowie unveränderte
47 positive und 319 negative Prüfclips, darunter 14 Satelliten-Negative.
Das Ambient-Prüfaudio bleibt dasselbe. Kein Mining, Collector-Sync oder
Firmware-Rollout erfolgt während der Vergleiche.

Einfaches Entfernen der drei Quellen verändert auch die Sampling-Verteilung:
Die verbleibenden Gewichte werden normalisiert, also steigen die Anteile
eigener Aufnahmen und TTS. Deshalb gibt es zusätzlich einen Vergleich mit
Ersatz durch die bereits vorhandenen WAV-Sprach-/Hintergrundkorpora:
`speech_extra: 25`, `background_negative: 10` statt jeweils 5. Die wegfallende
HF-Gewichtsmasse 25 wird dadurch ersetzt; positive/negative Gesamtsampling-
Masse und Klassengewichte bleiben wie beim Ausgangsmodell. Das prüft den
Nutzen gegenüber diesen konkreten Ersatzkorpora.

Die Ergebnisse werden nach vollständiger Bewertung beider Seeds ergänzt.
Eine Aussage über reale Telefonate benötigt darüber hinaus weitere
bestätigte Collector-Negative und einen unabhängigen Prüfbestand.
