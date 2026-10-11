# Installation

## Requirements

- Linux or macOS, Python 3.10–3.12.
- For training a GPU is strongly recommended (NVIDIA with CUDA, or AMD with
  ROCm builds of TensorFlow). A CPU works but takes hours for a real model.
- About 15 GB of disk for the shared downloads.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "wake-word-trainer[train,tts] @ git+https://github.com/ChristophBellmann/wake-word-trainer"
```

| Extra | Brings | |
| --- | --- | --- |
| `train` | TensorFlow, microWakeWord | training |
| `tts` | Piper, ONNX Runtime | synthetic speech (runs on the CPU) |
| `segment` | faster-whisper | [clip extraction](extraction.md) for the collector |
| `dev` | pytest, ruff | development |

Without `tts` you can still train from your own recordings.

microWakeWord is pinned to a fixed upstream version, installed from the
branch [`trainer-compat`](https://github.com/ChristophBellmann/micro-wake-word/tree/trainer-compat),
which adds only small fixes: the `__init__.py` files a normal install of
upstream misses, NumPy 1.26 compatibility, and two TensorFlow 2.20/ROCm export
fixes.

## AMD GPUs (ROCm)

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

The constraints file keeps TensorFlow, NumPy and protobuf, also when
TensorFlow came from a local wheel. Speech generation uses Piper directly and
needs neither PyTorch nor the Piper Sample Generator with its NumPy 2
dependency; it is tested with NumPy 1.26.

For a systemd service, copy the ROCm library paths of a working terminal
training into the local service configuration.

## Check

```bash
wake-word-trainer --version
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
