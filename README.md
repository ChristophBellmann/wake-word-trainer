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
directly through Piper’s ONNX API.

[Deutsch weiter unten](#deutsch)

The full documentation is in [docs/](docs/README.md) (also as a [GitBook](https://renewable-energy-design.gitbook.io/wake-word-trainer/)):
[Installation](docs/installation.md) ·
[Quick start](docs/quick-start.md) ·
[Settings](docs/configuration.md) ·
[Pipeline](docs/pipeline.md) ·
[Evaluation](docs/evaluation.md) ·
[Home Assistant service](docs/service.md) ·
[Automatic deployment](docs/deployment.md) ·
[Idle-night training](docs/automatic-training.md) ·
[Command line](docs/cli.md) ·
[HTTP API](docs/api.md)

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

`train` brings TensorFlow and microWakeWord, `tts` brings Piper and ONNX Runtime.
Synthetic speech runs on the CPU; training uses the configured TensorFlow GPU.
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
python - <<'PYTHON'
from importlib.metadata import version
from pathlib import Path
Path("keep.txt").write_text("".join(f"{name}=={version(name)}\n" for name in ("tensorflow", "numpy", "protobuf")))
PYTHON
pip install -c keep.txt "wake-word-trainer[train,tts] @ git+https://github.com/ChristophBellmann/wake-word-trainer"
```

Die Constraints-Datei hält TensorFlow, NumPy und protobuf fest, auch wenn
TensorFlow aus einer lokalen Wheel-Datei installiert wurde. Die
Sprachgenerierung verwendet Piper direkt und braucht weder PyTorch noch den
Piper Sample Generator mit dessen NumPy-2-Abhängigkeit. Das ist auch mit
NumPy 1.26 geprüft. Bei systemd die ROCm-Bibliothekspfade des funktionierenden
Terminal-Trainings in die lokale Dienstkonfiguration übernehmen.

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
  seed: null                 # optional integer: augmentation, sampling and model initialization
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

### VRAM mit anderen Diensten teilen

Wenn auf dem Trainingsrechner auch ein LLM oder GPU-TTS läuft, können die
systemd-Benutzerdienste in `service.yaml` eingetragen werden:

```yaml
pause_services:
  - llama-server.service
  - orpheus-llm.service
  - wyoming-orpheus.service
```

Vor einem Lauf hält der Trainer die zuvor aktiven Dienste an. Nach Abschluss,
Fehler, „Training stoppen“ oder regulärem Beenden des Trainer-Dienstes startet
er genau diese Dienste wieder. Zuvor inaktive Dienste bleiben inaktiv. Ein
unterbrochener Vorgang wird beim nächsten Trainer-Start wiederhergestellt;
der Zustand liegt lokal im Projekt unter `paused_services.json`. Die Namen
sind Beispiele; ohne diese Einstellung wird kein anderer Dienst verwaltet.
Dies gilt für `serve`; direkte CLI-Läufe verwalten keine fremden Dienste.

### Freie Systemressourcen vor und während eines Laufs

Pausierte Dienste machen VRAM frei, aber nicht den Arbeitsspeicher, den ein
anderer Prozess belegt. Ein Training, das auslagern muss, kommt kaum voran.
`serve` prüft deshalb vor jedem Start (aus Home Assistant, über die API und für
das nächtliche Training) und lehnt mit HTTP 503 und dem Grund ab:

```yaml
resource_check:                 # alle Werte optional
  min_available_memory_gb: 6    # MemAvailable vor und während des Laufs
  max_load_per_cpu: 0.75        # Last pro CPU vor dem Start
  min_free_vram_gb: 8           # nach dem Pausieren der Dienste; ohne Angabe nicht geprüft
  wait_seconds: 10              # Zeit, bis freigegebener VRAM sichtbar ist
```

Abgelehnte Starts verändern den Zustand des letzten Laufs nicht. Scheitert
die VRAM-Prüfung nach dem Pausieren, startet der Trainer die Dienste wieder.
Während eines Laufs prüft er den Arbeitsspeicher alle 30 Sekunden.
`/v1/status` zeigt unter `resources` die Messwerte und eine Warnung, die auch
in `service.log` steht. Die Warnung bricht den Lauf nicht ab.
`enabled: false` schaltet die Prüfung ab.

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

## GitBook source and maintenance

The [project documentation](https://renewable-energy-design.gitbook.io/wake-word-trainer/) is published as a GitBook through GitSync from
`ChristophBellmann/wake-word-trainer` on `main`. Page sources live in
[`docs/`](docs/README.md), with navigation in
[`docs/SUMMARY.md`](docs/SUMMARY.md). `.gitbook.yaml` and
`gitbook-docs.yaml` point to the same content directory.

Update the affected pages alongside changes to features, installation,
configuration, the HTTP API or verification results. Add new pages to
`SUMMARY.md`, commit and push to `main`, then verify the published GitBook:
synchronization is asynchronous.

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

Schwierige Beispiele dürfen den Bewertungsbestand nicht ins Training
zurückbringen: Liegt derselbe WAV-Inhalt auch unter `recordings.hard_folders`,
bleibt eine bereits zur Bewertung zugeordnete Aufnahme ausschließlich dort.
`fetch` entfernt frühere doppelte Trainingskopien; Quelldateien bleiben erhalten.

## Automatic wake-word clips from long recordings

Install the optional `segment` extra (`pip install ".[segment]"`, using your
existing constraints on ROCm) and enable `extraction` in the service YAML:

```yaml
extraction:
  enabled: true
  model: small
  cpu_threads: 4
  padding_ms: 200
  min_probability: 0.35
```

The Collector sends a manual recording to authenticated `POST /v1/extract`
(WAV body, `X-Wakeword-Phrases` JSON list of its configured phrase and variants).
Local faster-whisper CPU/int8 recognition produces word timestamps. Use a current
model conversion with `alignment_heads` metadata (the default `small` download
provides it); older Wyoming model folders may lack word alignment support. The
service returns complete matching phrases with bounded padding; it does not
write recordings into the training project. Audio is processed in memory and is not stored on the workstation. Models may be downloaded once; audio is never sent to a cloud.
The project's `language` selects the recognition language. The model is
independent of the wake-word model being trained, so missed wake-word
activations do not exclude examples. Extraction runs serially on the CPU
without pausing GPU services or changing the active training run.

Requests are bounded to 4 MiB and 0.5–120 seconds of mono PCM16 WAV (16/48 kHz).
`/v1/status` advertises `extraction.enabled` and `extraction.available`.
Recognition is imperfect: difficult pronunciations may need additional
Collector variants or manual editing. Clips must be listened to and accepted
in the Collector before training. Speech regions are recognized separately so repeated identical phrases are
not collapsed by transcription. `silence_ms` (default 500) controls separation.
A second between repetitions helps cutting.

Deutsch: Mit dem Extra `segment` und `extraction.enabled: true` schneidet der
Collector längere Mikrofonaufnahmen automatisch in einzelne Aktivierungswörter.
Phrase und Varianten kommen aus der HA-Integration; Sprache und Erkennungsmodell
sind konfigurierbar. Das Original bleibt in HA, die Ausschnitte warten dort auf
Freigabe. Der bestehende Trainingslauf und dessen GPU bleiben unberührt.

### Automatic firmware deployment (local configuration)

Training completion alone does not update a satellite. To opt into automatic
OTA, add this to the private project's `wakeword.yaml`:

```yaml
deployment:
  enabled: true
  reference_model: deployed/model.tflite
  command: [python3, /path/to/model_update.py, --config, /path/to/wake-word-model.yaml,
            --expected-sha256, "{sha256}", --wait, "120", --build]
```

First copy the model currently running on the devices **and its adjacent
manifest** to `deployed/model.tflite` and `deployed/model.json`. Device paths,
network access, credentials and flash commands stay local; the HTTP client
cannot set them. The command may run over SSH on the ESPHome host. It must
verify every requested device and return nonzero on any failed deployment.
Use the Collector's public `model_update.py` with `require_parity: true` and
`{hash}` in its model filename.

The pipeline evaluates the new export and deployed reference on the same
held-out own recordings and background audio. It compares both at their
manifest thresholds. Automatic deployment requires no regression in recall or
own negative triggers, an improvement in at least one of them, and compliance
with the configured false activation budgets. Synthetic-only evaluation cannot
approve automatic OTA. The report includes a comparison bound to both model
SHA256 values before Home Assistant receives completion.

After training, the service restores the paused GPU services, then runs the
configured deployment command. `/v1/status` includes a separate `deployment`
state; failed firmware updates keep the completed training and exported model.
The deployed reference advances only when **all** devices have succeeded.
Retry with `wake-word-trainer -p <project> deploy`; to compare an already
completed export first, use `deploy --compare`. Local details are in
`deployment.json`, `parity.json` and `deployment.log`. The Collector tool keeps
per-device progress and skips successful devices on retries.

### Continuous improvement during idle nights

The service can automatically train new accepted positives and negatives:

```yaml
automatic_training:
  enabled: true
  start: "01:00"                 # workstation local time
  end: "06:00"
  idle_seconds: 1200             # no desktop input for 20 minutes
  idle_command: [/path/to/venv/bin/python, -m, wake_word_trainer.idle]
  min_new_samples: 5
  min_interval_hours: 20         # at most one automatic attempt per night
  profile: recommended
  max_gpu_use_percent: 5         # unknown GPU activity blocks automation
  max_load_per_cpu: 0.5
  check_seconds: 30
```

`wake_word_trainer.idle` asks GNOME's idle monitor over the session D-Bus
(X11 and Wayland) and otherwise measures the X11 screensaver's input
inactivity using libX11/libXss (Cinnamon, MATE, Xfce, KDE on X11). A systemd
user service does not inherit `DISPLAY` and `XAUTHORITY` from the desktop; the
probe takes them from the systemd user manager, `/tmp/.X11-unix` and
`~/.Xauthority`. A Wayland session never falls back to XWayland, which does
not see native input. On other desktops configure a command that prints idle
**seconds**, not a constant. An unavailable or invalid probe blocks the
automatic run; `/v1/status.automatic_training.error` then shows the probe's
reason (`idle probe: …`). GPU use and CPU load are checked before starting; use
`max_gpu_use_percent: null` for a CPU-only workstation.

The first check establishes a baseline of collector sample hashes. New data
arriving during training stays pending for the next run. A failed or interrupted
run does not consume examples. Cooldown prevents repeated failed attempts in
one night. When desktop input resumes or the night window closes, only the
automatically started run stops; GPU services are restored. Manual runs are
never interrupted by this policy. Existing firmware deployment blocks a new
training. `/v1/status.automatic_training` explains why the scheduler is waiting.

In the Collector enable **Learn from activations without input** and install
its matching ESPHome package. A confirmed empty STT result labels the buffered
activation audio as a negative; technical failures remain for review. The
normal held-out comparison still gates each automatic deployment. Training on
new negatives cannot guarantee an improvement: an equal or worse model is
kept out of the devices.
