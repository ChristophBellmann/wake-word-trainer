# Einstellungen (wakeword.yaml)

`init` erstellt im Projekt `wakeword.yaml`. Darin stehen nur Abweichungen
von den Defaults. Die folgende Übersicht zeigt die wichtigsten Defaults;
Schlüssel und Profilkennungen bleiben als API-Vertrag unverändert.

```yaml
wake_word: Hey Jarvis
slug: hey_jarvis              # Dateinamen; aus wake_word abgeleitet
language: en                  # Sprache im ESPHome-Manifest
collector:
  url: ""                     # Basis-URL von Home Assistant
  token_file: ""              # private Datei mit Collector-Token
recordings:
  folders: []                 # weitere freigegebene positive WAV-Ordner
  hard_folders: []            # schwierige echte Aufrufe; nur Training
  negative_folders: []        # bestätigte Aufnahmen ohne Aktivierungswort
  eval_share: 0.2             # getrennt gehaltener Prüfanteil
negatives:
  speech_folders: []          # weitere negative Sprache als 16-kHz-WAV
  background_folders: []      # negative WAVs; getrennt von Augmentierung
  speech_presets: []          # Download mit download --speech, etwa [mls_de]
  speech_clips: 6000          # höchstens so viele, gleichmäßig ausgewählt
  eval_share: 0.05            # je Anteil für Validierung und Prüfung externer WAVs
  feature_sets: [speech, dinner_party, no_speech, dinner_party_eval]
tts:
  voices: []                  # Piper-ONNX-Dateien mit benachbarter .onnx.json
  phrases: []                 # synthetisierte positive Aussprachevarianten
  samples: 2000
  negative_phrases: []        # ähnliche negative Phrasen
  negative_samples: 1000
  length_scales: [0.8, 0.9, 1.0, 1.1, 1.2]
  min_positive_duration_seconds: 0.0
augmentation:
  background_probability: 0.25
  background_min_snr_db: 6
  background_max_snr_db: 22
  rir_probability: 0.3
  augment_negatives: true     # false: keine Augmentierung der Trainingsnegative; Prüfung bleibt sauber
  background_folders: []      # eigene Raumgeräusche
  rir_folders: []
training:
  device: auto                # gpu verlangt GPU, cpu erzwingt CPU
  engine: standard            # optimized: Graph, Prefetch/TFRecord und Streaming-Validierung
  input_pipeline: prefetch
  tfrecord_examples: 65536
  input_threads: 4
  feature_workers: 1                 # CPU-Feature-Prozesse, 1 bis 32; spawn
  compute_threads: 4
  prefetch_batches: 2
  validation_cache_mb: 512
  validation_batch_size: 1024
  progress_interval: 25
  seed: null                 # optional: fester Seed für reproduzierbare Vergleiche
  model_profile: esphome_v2_like  # weitere Profile: current und current_wider
  time_mask_max_size: 0
  time_mask_count: 0
  freq_mask_max_size: 0
  freq_mask_count: 0
  steps: 20000                # je Runde
  batch_size: 128
  learning_rate: 0.001
  early_stopping_patience: 0  # vollständige Bewertungen ohne Verbesserung; null deaktiviert
  early_stopping_min_delta: 0.0
  lr_patience: 0              # Bewertungen vor Lernratenabsenkung; null deaktiviert
  lr_factor: 0.5
  lr_max_reductions: 3
  min_learning_rate: 0.00001
  positive_class_weight: 1.0
  negative_class_weight: 20.0
  own_repeat: 8               # augmentierte Kopien je eigener Aufnahme
  weights: {own: 3.0, own_hard: 1.5, own_negative: 4.0, satellite_negative: 4.0, speech_extra: 5.0, mined: 4.0,
            background_negative: 5.0, tts: 2.0, tts_negative: 3.0, speech: 10.0, dinner_party: 10.0, no_speech: 5.0}
  clip_duration_ms: 1500
  eval_step_interval: 500
  mining_rounds: 0            # zusätzliche Mining- und Trainingsrunden
  mining_samples: 20000       # negative spectrograms checked je Runde
  mining_threshold: 0.4
evaluation:
  max_false_accepts_per_hour: 0.5
  min_probability_cutoff: 0.5
  max_own_negative_share: 0.05  # Anteil eigener Prüfnegative
  sliding_window_size: 5
export:
  author: ""
  website: ""
  tensor_arena_size: 30000    # erhöhen, falls das Gerät das Modell nicht laden kann
  minimum_esphome_version: 2024.7.0
```

## Wichtige Entscheidungen

Quellgewichte bestimmen Sampling-Anteile unabhängig von Dateianzahlen.
`recordings.hard_folders` enthält schwierige echte Aufrufe nur fürs Training.
Die getrennten eigenen Prüfclips bleiben unverändert. Externe negative WAVs
haben zusätzliche Validierungs- und Prüfanteile.

`training.device: gpu` verlangt tatsächliche GPU-Ausführung. `optimized`
aktiviert den Graph-Kern mit begrenzter Zuführung und vollständiger Validierung.
Batchgröße und Poolgröße werden nach [Durchsatz und Qualität](gpu-training-and-benchmarks.md)
getrennt verglichen. `steps` und Frühstopp begrenzen den Lauf; Mining ist ein
zusätzliches Experiment mit negativen Trainingsdaten.

[Negativkorpora](negative-corpora.md) erklären passende 10-ms-Features und
Standardanteile. `--downloads <ordner>` teilt Downloads zwischen Projekten.
Projektaufnahmen, erzeugte Sprache, Features, Checkpoints und Berichte bleiben
privat. Die [Dienstkonfiguration](service.md) enthält HA-Profile und lokale
Ressourcenregeln; [Rollout](deployment.md) enthält die separate Freigabe.
