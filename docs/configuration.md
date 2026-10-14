# Settings (wakeword.yaml)

Every project folder has a `wakeword.yaml`, created by `init`. It only
contains what differs from these defaults:

```yaml
wake_word: Hey Jarvis
slug: hey_jarvis              # file names; derived from wake_word
language: en                  # written into the ESPHome manifest
collector:
  url: ""                     # Home Assistant base URL
  token_file: ""              # file with the collector token
recordings:
  folders: []                 # more WAV folders with the wake word
  hard_folders: []            # difficult but correct examples, extra weight, training only
  negative_folders: []        # own recordings WITHOUT the wake word (real false activations)
  eval_share: 0.2             # share held out for the evaluation
negatives:
  speech_folders: []          # more speech the model must ignore (16 kHz WAVs)
  background_folders: []      # AudioSet/FMA WAVs as negative examples, separate from augmentation
  speech_presets: []          # downloaded with `download --speech`, e.g. [mls_de]
  speech_clips: 6000          # at most this many, evenly picked
  eval_share: 0.05            # each: validation and testing for extra negative WAVs
  feature_sets: [speech, dinner_party, no_speech, dinner_party_eval]
tts:
  voices: []                  # Piper .onnx files (with .onnx.json beside them)
  phrases: []                 # what is synthesized; phonetic spellings help
  samples: 2000
  negative_phrases: []        # look-alikes the model must ignore
  negative_samples: 1000
  length_scales: [0.8, 0.9, 1.0, 1.1, 1.2]
  min_positive_duration_seconds: 0.0
augmentation:
  background_probability: 0.25
  background_min_snr_db: 6
  background_max_snr_db: 22
  rir_probability: 0.3
  augment_negatives: true     # false: keep negative training audio clean; evaluation is always clean
  background_folders: []      # e.g. recordings of your own living room
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
  seed: null                 # optional integer: augmentation, sampling and model initialization
  model_profile: esphome_v2_like  # also current and current_wider from the old lab
  time_mask_max_size: 0
  time_mask_count: 0
  freq_mask_max_size: 0
  freq_mask_count: 0
  steps: 20000                # per round
  batch_size: 128
  learning_rate: 0.001
  early_stopping_patience: 0  # full evaluations without improvement; zero disables
  early_stopping_min_delta: 0.0
  lr_patience: 0              # full evaluations before reducing LR; zero disables
  lr_factor: 0.5
  lr_max_reductions: 3
  min_learning_rate: 0.00001
  positive_class_weight: 1.0
  negative_class_weight: 20.0
  own_repeat: 8               # augmented copies of each own recording
  weights: {own: 3.0, own_hard: 1.5, own_negative: 4.0, satellite_negative: 4.0, speech_extra: 5.0, mined: 4.0,
            background_negative: 5.0, tts: 2.0, tts_negative: 3.0, speech: 10.0, dinner_party: 10.0, no_speech: 5.0}
  clip_duration_ms: 1500
  eval_step_interval: 500
  mining_rounds: 0            # extra rounds: mine false activations, train again
  mining_samples: 20000       # negative spectrograms checked per round
  mining_threshold: 0.4
evaluation:
  max_false_accepts_per_hour: 0.5
  min_probability_cutoff: 0.5
  max_own_negative_share: 0.05  # of your held-out non-wake-word recordings
  sliding_window_size: 5
export:
  author: ""
  website: ""
  tensor_arena_size: 30000    # raise it if the satellite cannot load the model
  minimum_esphome_version: 2024.7.0
```

Die [Negativkorpus-Inventur](negative-corpora.md) erklärt aktive Auswahl,
Sampling-Gewichte und den Vergleich mit Ersatzkorpora. Die drei alten
`*_background`-Archive besitzen 20-ms-Features und dürfen nicht mit den
10-ms-Features des aktuellen Trainers gemischt werden.

`--downloads <folder>` on the command line shares the downloads between
several projects.

## The most important settings

| Setting | |
| --- | --- |
| `tts.voices`, `tts.phrases` | synthetic positives; several voices and phonetic spellings help |
| `tts.negative_phrases` | look-alikes the model must ignore ("hey travis" for "hey jarvis") |
| `recordings.negative_folders` | your own recordings without the wake word: the strongest lever against false activations |
| `recordings.hard_folders` | difficult but correct examples, extra weight, training only |
| `recordings.eval_share` | share of your own recordings held out for evaluation |
| `negatives.speech_presets` | speech in your language, e.g. `[mls_de]` |
| `training.steps`, `training.mining_rounds` | length and number of training rounds |
| `evaluation.max_false_accepts_per_hour` | the budget that picks the cutoff |
| `export.tensor_arena_size` | raise it if the satellite cannot load the model |

The project folder also holds the fetched clips, generated speech, features,
checkpoints, `report.json` and `export/`.
