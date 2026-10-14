"""GPU-Auswahl, ROCm-Umgebung und gemeinsame Belegung für CLI und Dienst."""

from __future__ import annotations

import contextlib
import json
import os
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .project import ProjectError


def _python_rocm_sdk() -> bool:
    """Detect AMD's self-contained SDK without importing or initializing a GPU."""
    try:
        version("rocm-sdk-core")
        version("rocm-sdk-libraries")
    except PackageNotFoundError:
        return False
    return True


def environment(original: dict | None = None) -> dict:
    env = dict(os.environ if original is None else original)
    root = Path(env.get("ROCM_PATH", "/opt/rocm"))
    # TheRock's Python packages initialize their matching libraries themselves.
    # Injecting /opt/rocm or its RCCL into them would mix two different stacks.
    python_sdk = not env.get("ROCM_PATH") and _python_rocm_sdk()
    if root.is_dir() and not python_sdk:
        env.setdefault("ROCM_PATH", str(root))
        for key, paths in {
            "PATH": [root / "bin", root / "llvm/bin"],
            "LD_LIBRARY_PATH": [root / "lib", root / "lib64", root / "lib/llvm/lib", root / "lib/rocm_sysdeps/lib"],
            "LD_PRELOAD": [root / "lib/librccl.so.1", Path("/lib/x86_64-linux-gnu/libstdc++.so.6")],
        }.items():
            parts = [str(path) for path in paths if path.exists()] + env.get(key, "").split(":")
            env[key] = ":".join(dict.fromkeys(part for part in parts if part))
        env.setdefault("TF_ROCM_DISABLE_HIPBLASLT", "1")
        env.setdefault("TF_ROCM_USE_HIPBLASLT", "0")
        env.setdefault("TF_ROCM_DISABLE_HIPBLASLT_INIT", "1")
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def lock_path() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "wake-word-trainer/gpu.lock"


@contextlib.contextmanager
def lease(label: str, path: Path | None = None, inherited_fd: int | None = None):
    """Eine GPU-Arbeit pro Benutzer; kein fork und keine parallelen HIP-Initialisierungen."""
    import fcntl

    target = path or lock_path()
    if inherited_fd is not None:
        expected, actual = target.stat(), os.fstat(inherited_fd)
        if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
            raise ProjectError("Geerbte GPU-Belegung gehört nicht zur gemeinsamen Sperrdatei")
        fcntl.flock(inherited_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield None  # Der Elternprozess hält und beendet die Belegung.
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+") as handle:
        target.chmod(0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as err:
            raise ProjectError("GPU wird bereits von einem Trainer-, Teacher- oder Benchmark-Lauf belegt") from err
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "job": label, "started_at": time.time()}, handle)
        handle.flush()
        try:
            yield handle
        finally:
            handle.seek(0)
            handle.truncate()
            fcntl.flock(handle, fcntl.LOCK_UN)


def busy(path: Path | None = None) -> bool:
    import fcntl

    target = path or lock_path()
    if not target.exists():
        return False
    with target.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def tensorflow_device(request: str = "auto", threads: int = 4) -> dict:
    import tensorflow as tf

    if request not in {"auto", "gpu", "cpu"}:
        raise ProjectError("Gerät muss auto, gpu oder cpu sein")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    gpus = tf.config.list_physical_devices("GPU")
    if request == "gpu" and not gpus:
        raise ProjectError("GPU angefordert, aber TensorFlow erkennt keine GPU; ROCm/CUDA-Umgebung prüfen")
    if request == "cpu":
        tf.config.set_visible_devices([], "GPU")
        gpus = []
    else:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    selected = "/GPU:0" if gpus else "/CPU:0"
    with tf.device(selected):
        probe = tf.matmul(tf.ones((16, 16)), tf.ones((16, 16)))
    probe.numpy()  # Ausführung abschließen: Geräteerkennung allein ist kein Beweis.
    if selected == "/GPU:0" and "GPU" not in probe.device:
        raise ProjectError("TensorFlow hat die angeforderte GPU-Operation auf CPU ausgeführt")
    return {
        "framework": "tensorflow",
        "version": tf.__version__,
        "device": selected,
        "operation_device": probe.device,
        "gpu": tf.config.experimental.get_device_details(gpus[0]).get("device_name", "") if gpus else "",
        "rocm": bool(tf.sysconfig.get_build_info().get("is_rocm_build")),
        "threads": threads,
    }


def torch_device(request: str = "auto"):
    import torch

    if request not in {"auto", "gpu", "cpu"}:
        raise ProjectError("Gerät muss auto, gpu oder cpu sein")
    available = request != "cpu" and torch.cuda.is_available()
    if request == "gpu" and not available:
        raise ProjectError("GPU angefordert, aber PyTorch erkennt keine GPU; ROCm/CUDA-Wheel und Umgebung prüfen")
    return torch.device("cuda:0" if available and request != "cpu" else "cpu")


def snapshot(drm: Path = Path("/sys/class/drm")) -> dict:
    cards = []
    for device in drm.glob("card*/device"):
        try:
            total = int((device / "mem_info_vram_total").read_text())
            used = int((device / "mem_info_vram_used").read_text())
            use = float((device / "gpu_busy_percent").read_text())
        except (OSError, ValueError):
            continue
        if total <= 0:
            continue
        row = {"use_percent": use, "memory_percent": 100 * used / total, "vram_used_bytes": used}
        for path in device.glob("hwmon/hwmon*/power1_average"):
            try:
                row["power_watts"] = int(path.read_text()) / 1e6
            except (OSError, ValueError):
                pass
        cards.append((total, row))
    return max(cards, key=lambda row: row[0])[1] if cards else {}
