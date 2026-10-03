"""Reading and writing WAV files as 16 kHz mono float32, without heavy dependencies."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

RATE = 16000


def read(path: str | Path) -> np.ndarray:
    """Any PCM/float WAV -> mono float32 in [-1, 1] at 16 kHz."""
    rate, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != RATE:
        divisor = gcd(int(rate), RATE)
        audio = resample_poly(audio, RATE // divisor, int(rate) // divisor).astype(np.float32)
    return audio


def write(path: str | Path, audio: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(audio, dtype=np.float32) * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(str(path), RATE, pcm)


def wavs(*folders: str | Path) -> list[Path]:
    """All WAV files below the folders, sorted for reproducible runs."""
    found: list[Path] = []
    for folder in folders:
        folder = Path(folder)
        if folder.is_dir():
            found.extend(path for path in folder.rglob("*.wav") if path.is_file())
    return sorted(found)
