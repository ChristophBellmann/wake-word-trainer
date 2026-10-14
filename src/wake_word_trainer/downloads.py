"""Shared data from Hugging Face: negative features, room impulse responses,
background audio and Piper voices. Downloaded once into <project>/downloads
(or a folder shared between projects, see --downloads).

Licenses differ per dataset (AudioSet, Free Music Archive, MIT IR survey,
the microWakeWord negative sets): treat models trained with them as suitable
for personal, non-commercial use, as the microWakeWord documentation does.
"""

from __future__ import annotations

import hashlib
import json
import re
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
ALL_NEGATIVE_SETS = (*NEGATIVE_SETS, "speech_background", "dinner_party_background", "no_speech_background")
# The legacy *_background archives use 20 ms; the ESPHome-compatible archives
# and this trainer use 10 ms. Downloadability does not imply compatibility.
LEGACY_BACKGROUND_SETS = ("speech_background", "dinner_party_background", "no_speech_background")
NEGATIVE_REPO = "kahrendt/microwakeword"
RIR_REPO = "davidscripka/MIT_environmental_impulse_responses"
RIR_FOLDER = "16khz"
AUDIOSET = ("agkphysics/AudioSet", "data/bal_train09.tar")
FMA = ("mchl914/fma_xsmall", "fma_xs.zip")
VOICES_REPO = "rhasspy/piper-voices"

# Speech the model must ignore, in other languages than the microWakeWord sets
# (mostly English). Multilingual LibriSpeech: read audio books, CC BY 4.0.
# A preset is a dataset repository plus words that select its Parquet files.
SPEECH_PRESETS = {
    f"mls_{code}": ("facebook/multilingual_librispeech", (language, "train"))
    for code, language in {
        "de": "german",
        "fr": "french",
        "nl": "dutch",
        "es": "spanish",
        "it": "italian",
        "pt": "portuguese",
        "pl": "polish",
    }.items()
}
SPEECH_PRESETS.update(
    {
        "voxpopuli_de": ("facebook/voxpopuli", ("de/train-",)),
        "voxpopuli_en": ("facebook/voxpopuli", ("en/train-",)),
        "librispeech_en": ("openslr/librispeech_asr", ("clean/train.100/",)),
    }
)

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


def _tree(repo: str, folder: str = "", base: str = HF) -> list[str]:
    """All file paths of a dataset repository (below folder), following pagination."""
    url = f"{base}/api/datasets/{repo}/tree/main/{urllib.parse.quote(folder)}".rstrip("/") + "?recursive=true"
    paths: list[str] = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            paths += [item["path"] for item in json.load(response) if item.get("type", "file") == "file"]
            link = response.headers.get("Link", "")
        url = ""
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
    return paths


def _dataset_file(repo: str, path: str, base: str = HF) -> str:
    return f"{base}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"


def _done(folder: Path) -> bool:
    return (folder / ".complete").is_file()


def _mark_done(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".complete").write_text("ok\n")


def negative_features(downloads: Path, log: Log = print, base: str = HF, sets=NEGATIVE_SETS) -> Path:
    """Pre-computed spectrograms of speech, music and noise (microWakeWord)."""
    target = downloads / "negative_datasets"
    for name in sets:
        if name not in ALL_NEGATIVE_SETS:
            raise ValueError(f"Unknown negative dataset: {name}")
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
    files = [path for path in _tree(RIR_REPO, RIR_FOLDER, base) if path.endswith(".wav")]
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


def speech(preset: str, downloads: Path, clips: int, log: Log = print, base: str = HF, exclude_phrases=()) -> Path:
    """Up to `clips` utterances of a speech preset as 16 kHz WAVs in downloads/speech/<preset>."""
    if preset not in SPEECH_PRESETS:
        raise ValueError(f"Unknown speech preset {preset!r}; known: {', '.join(sorted(SPEECH_PRESETS))}")
    repo, words = SPEECH_PRESETS[preset]
    target = downloads / "speech" / preset
    signature = {"clips": clips, "exclude_phrases": sorted(exclude_phrases), "version": 2}
    stamp = target / ".complete"
    if stamp.is_file() and stamp.read_text() == json.dumps(signature, sort_keys=True):
        return target
    if len(audio.wavs(target)) >= clips and not exclude_phrases:
        return target
    if stamp.is_file() and exclude_phrases:
        try:
            previous = json.loads(stamp.read_text())
        except ValueError:
            previous = {}
        if previous.get("exclude_phrases") != signature["exclude_phrases"] or previous.get("version") != 2:
            raise ValueError(f"Speech source {preset}: transcript filter differs or is unverified; use a new folder")
    files = sorted(
        path for path in _tree(repo, "", base) if path.endswith(".parquet") and all(word in path for word in words)
    )
    if not files:
        raise ValueError(f"No Parquet files for {preset} in {repo}")
    count = len(audio.wavs(target))
    for path in files:
        if count >= clips:
            break
        archive = _fetch(_dataset_file(repo, path, base), downloads / "archives" / repo.replace("/", "_") / path, log)
        count = _parquet_to_wav(archive, target, count, clips, exclude_phrases)
        archive.unlink()
    target.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(signature, sort_keys=True))
    return target


def _parquet_to_wav(path: Path, target: Path, count: int, limit: int, exclude_phrases=()) -> int:
    import io

    import pyarrow.parquet as pq  # comes with the train extra (datasets)
    import soundfile

    target.mkdir(parents=True, exist_ok=True)
    fields = pq.read_schema(path).names
    text_keys = [key for key in ("text", "transcript", "sentence", "normalized_text", "raw_text") if key in fields]
    table = pq.read_table(str(path), columns=["audio", *text_keys])

    def words(text):
        return re.findall(r"[^\W_]+", text.casefold())

    blocked = [words(phrase) for phrase in exclude_phrases]
    for row in table.to_pylist():
        if count >= limit:
            break
        tokens = words(" ".join(str(row.get(key) or "") for key in text_keys))
        if any(phrase and any(tokens[i : i + len(phrase)] == phrase for i in range(len(tokens))) for phrase in blocked):
            continue
        cell = row["audio"]
        data = cell.get("bytes") if isinstance(cell, dict) else None
        if not data:
            continue
        name = hashlib.sha256(data).hexdigest() + ".wav"
        if (target / name).is_file():
            continue
        try:
            samples, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
        except Exception:  # a broken file in a large set is skipped
            continue
        mono = samples.mean(axis=1)
        audio.write(target / name, _resample(mono, rate))
        count += 1
    return count


def _resample(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == audio.RATE:
        return np.asarray(samples, dtype=np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(int(rate), audio.RATE)
    return resample_poly(samples, audio.RATE // divisor, int(rate) // divisor).astype(np.float32)


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
        audio.write(target / f"{path.stem}.wav", _resample(data.mean(axis=1), rate))
