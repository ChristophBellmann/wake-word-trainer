# The pipeline step by step

`run` does it all: `fetch`, `generate`, `features`, then training rounds
(round 1 from scratch, each further round mines first and continues), each
round evaluated on the same held-out data, and exports the best round. Every
step can also be run alone.

| Step | |
| --- | --- |
| `fetch` | Downloads the clips you accepted in the collector (and WAVs from your own folders). Each clip goes to *training* or *evaluation* by a hash of its content, so it keeps its side forever and the evaluation never sees training material. Clips you reject in the collector disappear here on the next fetch. |
| `generate` | Synthetic speech: the wake word and similar sounding phrases the model must ignore, in several voices and speeds. |
| `download` | Shared data from Hugging Face: background audio, room impulse responses, and pre-computed negative features (speech, music, noise). Once, then reused. |
| `features` | Augmented spectrograms: room echo, background noise, gain, EQ. Your own recordings are augmented several times and weigh most. |
| `train` | microWakeWord MixedNet, streaming, int8 quantized, the architecture of the official ESPHome models. Resumes after an interruption. |
| `evaluate` | Runs the finished model exactly like the satellite: recall on your held-out recordings, false activations per hour on hours of background audio, for all 256 possible cutoffs. |
| `export` | `<slug>.tflite` and the manifest `<slug>.json` for ESPHome. |
| `mine` | Runs the trained model over the negative *training* data and keeps what it wrongly reacts to; the next training learns from it. |
| `compare` | Several models on exactly the same held-out data. |

## Training and evaluation stay apart

The split by content hash is permanent: a clip that was once in evaluation
stays there, also after re-recording sessions and new fetches. A WAV that also
lies under `recordings.hard_folders` stays in evaluation only; `fetch`
removes earlier duplicate training copies, source files stay untouched.

## Options of `run`

| Option | |
| --- | --- |
| `--rounds N` | training rounds after the first (mining in between) |
| `--steps N` | steps per round (default `training.steps`) |
| `--resume` | continue from the last checkpoint instead of from scratch |
| `--force` | regenerate synthetic speech |
| `--prepare-only` | only fetch, generate and build features (checks the data) |

`status` shows the progress of the current or last run.
