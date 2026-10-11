# Development

```bash
git clone https://github.com/ChristophBellmann/wake-word-trainer
cd wake-word-trainer
pip install -e ".[train,dev]"
pytest -m "not slow"   # seconds
pytest -m slow         # trains a tiny model on synthetic data, about a minute
ruff check .
```

## Layout

| Module | |
| --- | --- |
| `cli.py` | command line |
| `project.py` | project folder and `wakeword.yaml` |
| `recordings.py` | fetch from the collector, hash split |
| `downloads.py` | shared data, Piper voices, speech presets |
| `piper_generate.py`, `tts.py` | synthetic speech |
| `features.py`, `audio.py` | augmentation and spectrograms |
| `training.py`, `training_worker.py` | microWakeWord training |
| `mining.py` | false-activation mining |
| `evaluate.py` | streaming evaluation and cutoff |
| `export.py` | ESPHome model and manifest |
| `pipeline.py`, `state.py` | `run`, progress state |
| `service.py` | HTTP service |
| `system_check.py`, `resources.py` | resource checks, paused services |
| `extraction.py` | clip extraction with faster-whisper |
| `deployment.py` | firmware rollout gate |
| `scheduler.py`, `idle.py` | idle-night training |

Every version gets a tag `vX.Y.Z` and a GitHub release. The documentation
lives in `docs/` and is published through GitBook; update the affected pages
with the change.
