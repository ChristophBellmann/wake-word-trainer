"""Shared data from Hugging Face: negative features, room impulse responses,
background audio and Piper voices. Downloaded once into <project>/downloads
(or a folder shared between projects, see --downloads).

Licenses differ per dataset (AudioSet, Free Music Archive, MIT IR survey,
the microWakeWord negative sets): treat models trained with them as suitable
for personal, non-commercial use, as the microWakeWord documentation does.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import audio

HF = "https://huggingface.co"

NEGATIVE_SETS = ("speech", "dinner_party", "no_speech", "dinner_party_eval")
NEGATIVE_REPO = "kahrendt/microwakeword"
RIR_REPO = "davidscripka/MIT_environmental_impulse_responses"
RIR_FOLDER = "16khz"
AUDIOSET = ("agkphysics/AudioSet", "data/bal_train09.tar")
FMA = ("mchl914/fma_xsmall", "fma_xs.zip")
VOICES_REPO = "rhasspy/piper-voices"

Log = Callable[[str], None]


def _fetch(url: str, target: Path, log: Log) -> Path:
    """Download to target (via target.part, so an interrupted download restarts)."""
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    log(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    part.rename(target)
    return target


def _dataset_file(repo: str, path: str, base: str = HF) -> str:
    return f"{base}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"


def _done(folder: Path) -> bool:
    return (folder / ".complete").is_file()


def _mark_done(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".complete").write_text("ok\n")


def negative_features(downloads: Path, log: Log = print, base: str = HF) -> Path:
    """Pre-computed spectrograms of speech, music and noise (microWakeWord)."""
    target = downloads / "negative_datasets"
    for name in NEGATIVE_SETS:
        if _done(target / name):
            continue
        archive = _fetch(_dataset_file(NEGATIVE_REPO, f"{name}.zip", base), downloads / "archives" / f"{name}.zip", log)
        with zipfile.ZipFile(archive) as zipped:
            _safe_extract_zip(zipped, target)
        _mark_done(target / name)
        archive.unlink()
    return target


def room_impulses(downloads: Path, log: Log = print, base: str = HF) -> Path:
    target = downloads / "rirs"
    if _done(target):
        return target
    listing_url = f"{base}/api/datasets/{RIR_REPO}/tree/main/{RIR_FOLDER}"
    with urllib.request.urlopen(listing_url, timeout=60) as response:
        files = [item["path"] for item in json.load(response) if item.get("path", "").endswith(".wav")]
    for path in files:
        _fetch(_dataset_file(RIR_REPO, path, base), target / Path(path).name, log)
    _mark_done(target)
    return target


def background(downloads: Path, log: Log = print, base: str = HF) -> Path:
    """One AudioSet part and the Free Music Archive "extra small" set, as 16 kHz WAVs."""
    target = downloads / "background"
    for name, (repo, path) in {"audioset": AUDIOSET, "fma": FMA}.items():
        folder = target / name
        if _done(folder):
            continue
        archive = _fetch(_dataset_file(repo, path, base), downloads / "archives" / Path(path).name, log)
        raw = downloads / "archives" / f"{name}_raw"
        if archive.suffix == ".tar":
            with tarfile.open(archive) as tar:
                tar.extractall(raw, filter="data")
        else:
            with zipfile.ZipFile(archive) as zipped:
                _safe_extract_zip(zipped, raw)
        _convert_to_wav(raw, folder, log)
        _mark_done(folder)
        shutil.rmtree(raw)
        archive.unlink()
    return target


def voice(name: str, folder: Path, log: Log = print, base: str = HF) -> Path:
    """A Piper voice by name, e.g. de_DE-thorsten-medium -> folder/<name>.onnx (+ .json)."""
    try:
        locale, speaker, quality = name.split("-", 2)
        language = locale.split("_")[0]
    except ValueError as err:
        raise ValueError(f"Voice names look like de_DE-thorsten-medium, not {name!r}") from err
    remote = f"{language}/{locale}/{speaker}/{quality}/{name}.onnx"
    url = f"{base}/{VOICES_REPO}/resolve/main/{remote}"
    _fetch(url + ".json", folder / f"{name}.onnx.json", log)
    return _fetch(url, folder / f"{name}.onnx", log)


def _safe_extract_zip(zipped: zipfile.ZipFile, target: Path) -> None:
    root = target.resolve()
    for member in zipped.infolist():
        destination = (target / member.filename).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"Unsafe path in archive: {member.filename}")
    zipped.extractall(target)


def _convert_to_wav(source: Path, target: Path, log: Log) -> None:
    import soundfile  # part of the train extra (via audiomentations)

    target.mkdir(parents=True, exist_ok=True)
    files = [path for path in sorted(source.rglob("*")) if path.suffix.lower() in {".flac", ".mp3", ".wav", ".ogg"}]
    log(f"Converting {len(files)} files to 16 kHz WAV")
    for path in files:
        try:
            data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        except Exception:  # a broken file in a large set is skipped
            continue
        mono = data.mean(axis=1)
        if rate != audio.RATE:
            from math import gcd

            from scipy.signal import resample_poly

            divisor = gcd(int(rate), audio.RATE)
            mono = resample_poly(mono, audio.RATE // divisor, int(rate) // divisor)
        audio.write(target / f"{path.stem}.wav", np.asarray(mono, dtype=np.float32))
