"""How well the streaming model works, measured the way the satellite runs it.

* Recall on your own held-out recordings (recordings/eval), each preceded by
  a second of silence and followed by half a second, as the device hears it.
* False activations per hour on long background recordings (dinner party
  noise, the microWakeWord evaluation set).

The device compares the mean of the last `sliding_window_size` probabilities
with `probability_cutoff`, both as 8-bit values; the cutoffs tried here are
exactly those 256 steps. The chosen cutoff is the lowest one that stays
within the false activation budget (the most sensitive allowed setting), but
not below evaluation.min_probability_cutoff.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import audio
from .features import STEP_MS
from .project import Project, ProjectError

STRIDE = 3
SLICE_SECONDS = STRIDE * STEP_MS / 1000
COOLDOWN_SLICES = 25  # after an activation the device ignores ~0.75 s
CUTOFFS = np.arange(256) / 255.0


@dataclass
class Report:
    wake_word: str
    model: str
    positives: int
    positive_source: str
    ambient_hours: float
    sliding_window_size: int
    max_false_accepts_per_hour: float
    probability_cutoff: float | None
    recall: float | None
    false_accepts_per_hour: float | None
    missed: list[str] = field(default_factory=list)
    curve: list[dict] = field(default_factory=list)
    # Own recordings without the wake word (held out): how many would wake the satellite.
    own_negatives: int = 0
    own_negatives_triggered: int | None = None
    triggered_by: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.probability_cutoff is None:
            return (
                f"No cutoff meets {self.max_false_accepts_per_hour} false activations per hour "
                "(and the limit for own non-wake-word recordings); train longer or with more negative data."
            )
        text = (
            f"cutoff {self.probability_cutoff:.3f}: recognizes {self.recall:.1%} of {self.positives} "
            f"{self.positive_source} recordings, {self.false_accepts_per_hour:.2f} false activations per hour "
            f"({self.ambient_hours:.1f} h background)"
        )
        if self.own_negatives:
            text += f"; {self.own_negatives_triggered} of {self.own_negatives} own non-wake-word recordings trigger"
        return text


def moving_average(probabilities: np.ndarray, window: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(probabilities) < window:
        return np.zeros(0)
    cumulative = np.concatenate([[0.0], np.cumsum(probabilities)])
    return (cumulative[window:] - cumulative[:-window]) / window


def false_accepts_per_hour(
    tracks: Iterable[np.ndarray], cutoffs: np.ndarray = CUTOFFS, slice_seconds: float = SLICE_SECONDS
) -> tuple[np.ndarray, float]:
    """Activations above each cutoff (with the device's cooldown) per hour of audio."""
    cooldown = max(1, round(COOLDOWN_SLICES * SLICE_SECONDS / slice_seconds))
    counts = np.zeros(len(cutoffs))
    slices = 0
    for track in tracks:
        track = np.asarray(track)
        slices += len(track)
        for index, cutoff in enumerate(cutoffs):
            above = np.flatnonzero(track > cutoff)
            next_allowed = -1
            for position in above:
                if position >= next_allowed:
                    counts[index] += 1
                    next_allowed = position + cooldown
            if len(above) == 0:
                break  # higher cutoffs cannot activate either
    hours = slices * slice_seconds / 3600
    if hours == 0:
        raise ProjectError("No background audio to measure false activations")
    return counts / hours, hours


def choose(
    faph: np.ndarray,
    budget: float,
    minimum: float = 0.0,
    negative_share: np.ndarray | None = None,
    max_negative_share: float = 1.0,
) -> int | None:
    """Index of the lowest (most sensitive) cutoff within the budget, not below minimum,
    and with at most max_negative_share of the own negative recordings triggering."""
    ok = (faph <= budget) & (CUTOFFS[: len(faph)] >= minimum - 1e-9)
    if negative_share is not None:
        ok &= negative_share <= max_negative_share + 1e-9
    allowed = np.flatnonzero(ok)
    return int(allowed[0]) if len(allowed) else None


def _positives(project: Project) -> tuple[list[Path], str]:
    own = audio.wavs(project.recordings / "eval")
    if own:
        return own, "own"
    from .features import _holdout

    _, _, test = _holdout(audio.wavs(project.path("tts", "positive")))
    if not test:
        raise ProjectError("No recordings to evaluate on: run fetch (or generate) first")
    return test, "synthetic"


def _ambient_tracks(downloads: Path, extra: Iterable[Path] = ()) -> list[np.ndarray]:
    from mmap_ninja.ragged import RaggedMmap

    roots = [downloads / "negative_datasets" / "dinner_party_eval" / "testing_ambient", *extra]
    tracks = []
    for root in roots:
        for mmap in sorted(Path(root).glob("**/*_mmap")):
            tracks.extend(np.asarray(item) for item in RaggedMmap(str(mmap)))
    if not tracks:
        raise ProjectError("Background set missing: run download first")
    return tracks


@dataclass
class Measurement:
    """Raw scores of one model on the evaluation data, before choosing a cutoff."""

    model: Path
    window: int
    positives: list[Path]
    positive_source: str
    positive_peaks: np.ndarray
    negatives: list[Path]
    negative_peaks: np.ndarray
    faph: np.ndarray
    hours: float

    def recall(self, index: int) -> float:
        return float((self.positive_peaks > CUTOFFS[index]).mean()) if len(self.positive_peaks) else 0.0

    def triggered(self, index: int) -> int:
        return int((self.negative_peaks > CUTOFFS[index]).sum())

    def negative_share(self) -> np.ndarray | None:
        if not len(self.negative_peaks):
            return None
        return np.array([(self.negative_peaks > cutoff).mean() for cutoff in CUTOFFS])

    def choose(self, settings: dict) -> int | None:
        return choose(
            self.faph,
            float(settings["max_false_accepts_per_hour"]),
            float(settings["min_probability_cutoff"]),
            self.negative_share(),
            float(settings["max_own_negative_share"]),
        )


def measure(project: Project, model_path: Path, downloads: Path, window: int, ambient_extra: Iterable[Path] = ()):
    from microwakeword.audio.audio_utils import generate_features_for_clip
    from microwakeword.inference import Model

    # A streaming model takes `stride` feature frames per inference: that is its input length.
    model = Model(str(model_path))
    slice_seconds = model.stride * STEP_MS / 1000

    def scores(spectrogram: np.ndarray) -> np.ndarray:
        return moving_average(np.asarray(model.predict_spectrogram(spectrogram)), window)

    def peak(path: Path) -> float:
        samples = audio.read(path)
        padded = np.concatenate([np.zeros(audio.RATE, np.float32), samples, np.zeros(audio.RATE // 2, np.float32)])
        averaged = scores(generate_features_for_clip(padded, STEP_MS))
        return float(averaged.max()) if len(averaged) else 0.0

    positives, source = _positives(project)
    negatives = audio.wavs(project.recordings / "negative" / "eval")
    ambient = [scores(track.astype(np.float32)) for track in _ambient_tracks(downloads, ambient_extra)]
    faph, hours = false_accepts_per_hour(ambient, slice_seconds=slice_seconds)
    return Measurement(
        model=model_path,
        window=window,
        positives=positives,
        positive_source=source,
        positive_peaks=np.array([peak(path) for path in positives]),
        negatives=negatives,
        negative_peaks=np.array([peak(path) for path in negatives]),
        faph=faph,
        hours=hours,
    )


def report(project: Project, m: Measurement, index: int | None, budget: float) -> Report:
    def rel(path: Path) -> str:
        return str(path.relative_to(project.root))

    result = Report(
        wake_word=project.config["wake_word"],
        model=str(m.model),
        positives=len(m.positives),
        positive_source=m.positive_source,
        ambient_hours=round(m.hours, 3),
        sliding_window_size=m.window,
        max_false_accepts_per_hour=budget,
        probability_cutoff=None if index is None else round(float(CUTOFFS[index]), 4),
        recall=None if index is None else round(m.recall(index), 4),
        false_accepts_per_hour=None if index is None else round(float(m.faph[index]), 3),
        curve=[
            {"cutoff": round(float(CUTOFFS[i]), 4), "recall": round(m.recall(i), 4), "faph": round(float(m.faph[i]), 3)}
            for i in range(0, 256, 5)
        ],
        own_negatives=len(m.negatives),
    )
    if index is not None:
        cutoff = CUTOFFS[index]
        result.missed = [rel(p) for p, s in zip(m.positives, m.positive_peaks, strict=True) if s <= cutoff]
        if m.negatives:
            result.own_negatives_triggered = m.triggered(index)
            result.triggered_by = [rel(p) for p, s in zip(m.negatives, m.negative_peaks, strict=True) if s > cutoff]
    return result


def evaluate(project: Project, model_path: Path, downloads: Path, ambient_extra: Iterable[Path] = ()) -> Report:
    """Evaluate the trained model and choose its cutoff; writes report.json."""
    settings = project.config["evaluation"]
    budget = float(settings["max_false_accepts_per_hour"])
    m = measure(project, model_path, downloads, int(settings["sliding_window_size"]), ambient_extra)
    result = report(project, m, m.choose(settings), budget)
    project.path("report.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def compare(project: Project, models: list[Path], downloads: Path) -> list[dict]:
    """Several models (e.g. the one in use and a new one) on exactly the same data.

    Each model is measured with its own manifest (<model>.json beside the
    .tflite, if present): its sliding window, and at its own cutoff as well as
    at the cutoff the budget allows. Writes compare.json."""
    settings = project.config["evaluation"]
    budget = float(settings["max_false_accepts_per_hour"])
    rows = []
    for model_path in models:
        manifest_path = model_path.with_suffix(".json")
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        micro = manifest.get("micro", {})
        window = int(micro.get("sliding_window_size", settings["sliding_window_size"]))
        m = measure(project, model_path, downloads, window)
        row = {"model": str(model_path), "budget": asdict(report(project, m, m.choose(settings), budget))}
        if "probability_cutoff" in micro:
            index = round(float(micro["probability_cutoff"]) * 255)
            row["own_cutoff"] = asdict(report(project, m, index, budget))
        rows.append(row)
    project.path("compare.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
