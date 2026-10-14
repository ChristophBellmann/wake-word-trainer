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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
pins = []
for name in ("tensorflow", "tensorflow-rocm", "numpy", "protobuf"):
    try:
        pins.append(f"{name}=={version(name)}\n")
    except PackageNotFoundError:
        pass
Path("keep.txt").write_text("".join(pins))
PYTHON
pip install --no-deps 'microwakeword @ git+https://github.com/ChristophBellmann/micro-wake-word@664a767db4474cd149d2d54038ab9144808951ef'
python - <<'PYTHON'
from importlib.metadata import requires
from packaging.requirements import Requirement
import subprocess, sys
# AMD's tensorflow-rocm provides the module, but not pip's tensorflow name.
deps = [r for r in requires("microwakeword")
        if Requirement(r).name.lower().replace("_", "-") != "tensorflow"]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-c", "keep.txt", *deps])
PYTHON
pip install -c keep.txt tensorboard "wake-word-trainer[tts] @ git+https://github.com/ChristophBellmann/wake-word-trainer"
```

The constraints file keeps the installed TensorFlow distribution, NumPy and
protobuf. AMD wheels can be named `tensorflow-rocm`: they provide the
`tensorflow` module, but pip does not consider them a distribution named
`tensorflow`. The commands therefore install microWakeWord without an
additional framework download, then resolve its other declared dependencies.
A `pip check` warning about microWakeWord's missing `tensorflow` distribution
can remain with an AMD wheel; verify the actual module and GPU operation.
Do not install a second TensorFlow distribution to silence that warning. Speech generation uses Piper directly and
needs neither PyTorch nor the Piper Sample Generator with its NumPy 2
dependency; it is tested with NumPy 1.26.

For a systemd service, copy the ROCm library paths of a working terminal
training into the local service configuration.

## Check

```bash
wake-word-trainer --version
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
