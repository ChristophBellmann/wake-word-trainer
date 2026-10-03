# Wake Word Trainer

Train a wake word model for ESPHome voice satellites (`micro_wake_word`) from
**your own recordings** plus synthetic speech, and get an honest measurement
of how well it works on your voice before you flash it.

It is the second half of
[Wake Word Collector](https://github.com/ChristophBellmann/ha-wake-word-collector):

```text
satellite ──► Wake Word Collector (Home Assistant) ──► Wake Word Trainer ──► satellite
  say it        record, sort, review, export             fetch, train,         micro_wake_word
                                                         evaluate, export
```

It also works without the collector, from folders of WAV files or synthetic
speech alone. Training uses [microWakeWord](https://github.com/kahrendt/microWakeWord)
(as a dependency) and [Piper](https://github.com/OHF-Voice/piper1-gpl) voices
through [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator).

[Deutsch weiter unten](#deutsch)

## What it does

| Step | |
| --- | --- |
| `fetch` | Downloads the clips you accepted in the collector (and WAVs from your own folders). Each clip goes to *training* or *evaluation* by a hash of its content, so it keeps its side forever and the evaluation never sees training material. Clips you reject in the collector disappear here on the next fetch. |
| `generate` | Synthetic speech: the wake word and similar sounding phrases the model must ignore, in several voices and speeds. |
| `download` | Shared data from Hugging Face: background audio, room impulse responses, and pre-computed negative features (speech, music, noise). Once, then reused. |
| `features` | Augmented spectrograms: room echo, background noise, gain, EQ. Your own recordings are augmented several times and weigh most. |
| `train` | microWakeWord MixedNet, streaming, int8 quantized, the architecture of the official ESPHome models. Resumes after an interruption. |
| `evaluate` | Runs the finished model exactly like the satellite: recall on your held-out recordings, false activations per hour on hours of background audio, for all 256 possible cutoffs. Picks the most sensitive cutoff within your false activation budget and lists the recordings it misses. |
| `export` | `<slug>.tflite` and the manifest `<slug>.json` for ESPHome. |
| `mine` | Runs the trained model over the negative *training* data and keeps what it wrongly reacts to; the next training learns from it. |
| `compare` | Several models (e.g. the one on your satellites and a new one) on exactly the same held-out data, each at its own cutoff and at the cutoff your budget allows. |
| `serve` | HTTP service for Home Assistant: start, stop and follow trainings, fetch the model, play test recordings to a satellite. |

`run` does it all: `fetch`, `generate`, `features`, then training rounds
(round 1 from scratch, each further round mines first and continues), each
round evaluated on the same held-out data, and exports the best round.

## Requirements

- Linux or macOS, Python 3.10–3.12.
- For training a GPU is strongly recommended (NVIDIA with CUDA, or AMD with
  ROCm builds of TensorFlow); a CPU works but takes hours for a real model.
- About 15 GB of disk for the shared downloads.

## Installation

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "wake-word-trainer[train,tts] @ git+https://github.com/ChristophBellmann/wake-word-trainer"
```

`train` brings TensorFlow and microWakeWord, `tts` brings PyTorch and Piper.
Without `tts` you can still train from your own recordings.

microWakeWord is pinned to a fixed upstream version, installed from the branch
[`trainer-compat`](https://github.com/ChristophBellmann/micro-wake-word/tree/trainer-compat)
which adds only small fixes: the `__init__.py` files a normal install of
upstream misses, NumPy 1.26 compatibility, and two TensorFlow 2.20/ROCm export
fixes.

### AMD GPUs (ROCm)

ROCm builds of TensorFlow come as their own wheels (often with `numpy<2`).
Install them first, then the trainer **without letting pip replace them**:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install /path/to/tensorflow_rocm*.whl            # your ROCm TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
pip freeze | grep -iE "^(tensorflow|numpy|protobuf)" > keep.txt
pip install -c keep.txt "wake-word-trainer[train,tts] @ git+https://github.com/ChristophBellmann/wake-word-trainer"
```

The constraints file keeps TensorFlow, NumPy and protobuf as they are; pip
fails instead of breaking the GPU stack if something does not fit.

## Quick start with the Wake Word Collector

1. In Home Assistant, *Wake Word Collector → Configure* shows the token. Save it:

   ```bash
   mkdir -p ~/.config/wake-word-trainer
   (umask 077; echo "<token>" > ~/.config/wake-word-trainer/hey_jarvis.token)
   ```

2. Create a project and download voices and data:

   ```bash
   wake-word-trainer -p hey_jarvis init "Hey Jarvis" --language en \
     --collector-url http://homeassistant.local:8123 \
     --token-file ~/.config/wake-word-trainer/hey_jarvis.token
   wake-word-trainer -p hey_jarvis download --voice en_US-lessac-medium --voice en_US-amy-medium --data
   ```

3. Edit `hey_jarvis/wakeword.yaml`: add the voices and similar sounding phrases:

   ```yaml
   tts:
     voices: [voices/en_US-lessac-medium.onnx, voices/en_US-amy-medium.onnx]
     phrases: [hey jarvis, hey jarviss]
     negative_phrases: [hey jarvo, hey travis, hey service, jarvis, hey]
   ```

4. Train, evaluate, export:

   ```bash
   wake-word-trainer -p hey_jarvis run
   ```

   The end of the output looks like this (numbers made up):

   ```text
   cutoff 0.906: recognizes 93.8% of 48 own recordings, 0.31 false activations per hour (8.3 h background)
   Not recognized (3): listen to them; mislabelled clips belong rejected in the collector
   ```

5. Put `export/hey_jarvis.json` and `export/hey_jarvis.tflite` where ESPHome
   finds them (beside the device YAML, or on a web server) and use it:

   ```yaml
   micro_wake_word:
     models:
       - model: hey_jarvis.json
   ```

Record more in the collector, then `run` again: only new clips are fetched,
the synthetic speech is kept, training starts over with the new data.

## Settings

`wakeword.yaml` only contains what differs from these defaults:

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
  speech_presets: []          # downloaded with `download --speech`, e.g. [mls_de]
  speech_clips: 6000          # at most this many, evenly picked
tts:
  voices: []                  # Piper .onnx files (with .onnx.json beside them)
  phrases: []                 # what is synthesized; phonetic spellings help
  samples: 2000
  negative_phrases: []        # look-alikes the model must ignore
  negative_samples: 1000
  length_scales: [0.8, 0.9, 1.0, 1.1, 1.2]
augmentation:
  background_probability: 0.25
  background_min_snr_db: 6
  background_max_snr_db: 22
  rir_probability: 0.3
  background_folders: []      # e.g. recordings of your own living room
  rir_folders: []
training:
  steps: 20000                # per round
  batch_size: 128
  learning_rate: 0.001
  positive_class_weight: 1.0
  negative_class_weight: 20.0
  own_repeat: 8               # augmented copies of each own recording
  weights: {own: 3.0, own_hard: 1.5, own_negative: 4.0, speech_extra: 5.0, mined: 4.0,
            tts: 2.0, tts_negative: 3.0, speech: 10.0, dinner_party: 10.0, no_speech: 5.0}
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

`--downloads <folder>` shares the downloads between several projects.

## More data, fewer false activations

- **Your own negatives** (`recordings.negative_folders`, or clips marked
  *not the wake word* in the collector) are the strongest lever against false
  activations at home: TV, conversations, words that sound alike. A held-out
  share also limits the cutoff (`max_own_negative_share`) and the report
  lists the ones that still trigger.
- **Speech in your language**: the microWakeWord negative sets are mostly
  English. `download --speech mls_de` fetches German audio books
  (Multilingual LibriSpeech; also `mls_fr`, `mls_nl`, `mls_es`, `mls_it`,
  `mls_pt`, `mls_pl`); add the preset to `negatives.speech_presets`.
- **Mining rounds** (`training.mining_rounds: 2` or `run --rounds 2`) let the
  model find its own weak spots in the negative training data.

## Home Assistant

`wake-word-trainer serve` runs on the training computer and lets Home
Assistant (the [Wake Word Collector](https://github.com/ChristophBellmann/ha-wake-word-collector)
integration) start and stop trainings, show progress and results, take over
the finished model and run a loudspeaker test against a satellite. Copy
[`deploy/service.example.yaml`](deploy/service.example.yaml) and
[`deploy/wake-word-trainer.service`](deploy/wake-word-trainer.service),
create a token (`openssl rand -hex 24`), then in the collector's options
enter `http://<training computer>:10701` and the token.

## Replacing a model you already use

Measure before you switch:

```bash
wake-word-trainer -p hey_jarvis compare --current --model /path/to/old_model.tflite
```

With the old manifest (`old_model.json`) beside the `.tflite`, both models are
measured on your held-out recordings at their own cutoffs and at the cutoff
your budget allows. Switch only when the new one recognizes at least as much
with no more false activations.

## Good to know

- **Your recordings decide.** Synthetic voices are cheap, but a model that is
  good on them is not necessarily good on you. That is why the evaluation only
  uses your own held-out recordings (synthetic speech only if you have none,
  and the report says so). 50 or more own recordings from the satellites,
  from several people and distances, make a real difference.
- **Listen to the misses.** The evaluation lists the recordings the model
  does not recognize. Often they are mislabelled: something else was said.
  Reject them in the collector and fetch again.
- **One model per wake word.** Two phrases in one model compete; train them
  separately.
- **Short words are hard.** Three or four syllables ("Hey Jarvis") work much
  better than one ("Jarvis").
- **The cutoff is a trade-off.** A lower `max_false_accepts_per_hour` gives a
  higher cutoff and fewer accidental activations, but more misses.

## Data licenses

The downloaded datasets (AudioSet, Free Music Archive, MIT impulse responses,
microWakeWord's negative sets) have different licenses and usage
restrictions. As microWakeWord states for the same data: models trained with
them are for personal, non-commercial use.

## Privacy

Your recordings stay in the project folder on your computer. Do not publish
projects or models trained on other people's voices without their consent.

## Development

```bash
pip install -e ".[train,dev]"
pytest -m "not slow"   # seconds
pytest -m slow         # trains a tiny model on synthetic data, about a minute
```

## License

Apache-2.0

---

## Deutsch

Ein Aktivierungswort-Modell für ESPHome-Sprachsatelliten trainieren, aus den
**eigenen Aufnahmen** (gesammelt mit dem
[Wake Word Collector](https://github.com/ChristophBellmann/ha-wake-word-collector))
und synthetischer Sprache, und vor dem Flashen ehrlich messen, wie gut es bei
der eigenen Stimme erkennt.

```bash
wake-word-trainer -p hey_jarvis init "Hey Jarvis" --language de \
  --collector-url http://homeassistant.local:8123 --token-file ~/.config/wake-word-trainer/hey_jarvis.token
wake-word-trainer -p hey_jarvis download --voice de_DE-thorsten-medium --voice de_DE-kerstin-low --data
# wakeword.yaml: tts.voices, tts.phrases, tts.negative_phrases eintragen
wake-word-trainer -p hey_jarvis run
```

Der Trainer holt die im Collector angenommenen Aufnahmen und teilt sie fest in
Training und Bewertung. Er erzeugt synthetische Beispiele und ähnlich klingende
Gegenbeispiele und trainiert mit microWakeWord. Bewertet wird nur auf den
zurückgehaltenen eigenen Aufnahmen. Die Schwelle wird nach dem erlaubten Budget
an Fehlauslösungen pro Stunde gewählt. Am Ende liegen `.tflite` und das
ESPHome-Manifest unter `export/`. Nicht erkannte Aufnahmen werden aufgelistet:
anhören und, wenn falsch, im Collector verwerfen.
