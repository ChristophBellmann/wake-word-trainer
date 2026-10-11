# Command line

```text
wake-word-trainer [-p PROJECT] [--downloads FOLDER] <command> [options]
```

`-p` is the project folder with `wakeword.yaml` (default: current folder),
`--downloads` a shared download folder (default: `<project>/downloads`).

| Command | |
| --- | --- |
| `init "<wake word>"` | create `wakeword.yaml`; `--language`, `--collector-url`, `--token-file` |
| `download` | shared data and Piper voices: `--voice <name>` (repeatable), `--speech <preset>` (e.g. `mls_de`), `--data` with them also the shared data |
| `fetch` | own recordings from the collector and `recordings.folders` |
| `generate` | synthetic speech with Piper; `--force` regenerates |
| `features` | augmented spectrograms |
| `train` | train the model; `--restart` ignores the last checkpoint |
| `evaluate` | recall on own recordings, false activations, cutoff |
| `export` | model and manifest for ESPHome |
| `run` | everything; `--rounds`, `--steps`, `--resume`, `--force`, `--prepare-only` |
| `compare` | several models on the same evaluation data; `--model <tflite>` (repeatable), `--current` |
| `mine` | collect what the trained model wrongly reacts to |
| `status` | progress of the current or last run |
| `deploy` | retry the configured firmware rollout; `--compare` compares first |
| `serve --config <yaml>` | HTTP service for Home Assistant |

`wake-word-trainer --version` prints the version.
