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

    def summary(self) -> str:
        if self.probability_cutoff is None:
            return (
                f"No cutoff meets {self.max_false_accepts_per_hour} false activations per hour; "
                "train longer or with more negative data."
            )
        return (
            f"cutoff {self.probability_cutoff:.3f}: recognizes {self.recall:.1%} of {self.positives} "
            f"{self.positive_source} recordings, {self.false_accepts_per_hour:.2f} false activations per hour "
            f"({self.ambient_hours:.1f} h background)"
        )


def moving_average(probabilities: np.ndarray, window: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(probabilities) < window:
        return np.zeros(0)
    cumulative = np.concatenate([[0.0], np.cumsum(probabilities)])
    return (cumulative[window:] - cumulative[:-window]) / window


def false_accepts_per_hour(tracks: Iterable[np.ndarray], cutoffs: np.ndarray = CUTOFFS) -> tuple[np.ndarray, float]:
    """Activations above each cutoff (with the device's cooldown) per hour of audio."""
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
                    next_allowed = position + COOLDOWN_SLICES
            if len(above) == 0:
                break  # higher cutoffs cannot activate either
    hours = slices * SLICE_SECONDS / 3600
    if hours == 0:
        raise ProjectError("No background audio to measure false activations")
    return counts / hours, hours


def choose(faph: np.ndarray, budget: float, minimum: float = 0.0) -> int | None:
    """Index of the lowest (most sensitive) cutoff within the budget, not below minimum."""
    allowed = np.flatnonzero((faph <= budget) & (CUTOFFS[: len(faph)] >= minimum - 1e-9))
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


def evaluate(project: Project, model_path: Path, downloads: Path, ambient_extra: Iterable[Path] = ()) -> Report:
    from microwakeword.audio.audio_utils import generate_features_for_clip
    from microwakeword.inference import Model

    window = int(project.config["evaluation"]["sliding_window_size"])
    budget = float(project.config["evaluation"]["max_false_accepts_per_hour"])
    model = Model(str(model_path), stride=STRIDE)

    def scores(spectrogram: np.ndarray) -> np.ndarray:
        return moving_average(np.asarray(model.predict_spectrogram(spectrogram)), window)

    positives, source = _positives(project)
    peaks = []
    for path in positives:
        samples = audio.read(path)
        padded = np.concatenate([np.zeros(audio.RATE, np.float32), samples, np.zeros(audio.RATE // 2, np.float32)])
        averaged = scores(generate_features_for_clip(padded, STEP_MS))
        peaks.append(float(averaged.max()) if len(averaged) else 0.0)
    peaks_array = np.asarray(peaks)
    recall = np.array([(peaks_array > cutoff).mean() for cutoff in CUTOFFS])

    ambient = [scores(track.astype(np.float32)) for track in _ambient_tracks(downloads, ambient_extra)]
    faph, hours = false_accepts_per_hour(ambient)

    chosen = choose(faph, budget, float(project.config["evaluation"]["min_probability_cutoff"]))
    report = Report(
        wake_word=project.config["wake_word"],
        model=str(model_path),
        positives=len(positives),
        positive_source=source,
        ambient_hours=round(hours, 3),
        sliding_window_size=window,
        max_false_accepts_per_hour=budget,
        probability_cutoff=None if chosen is None else round(float(CUTOFFS[chosen]), 4),
        recall=None if chosen is None else round(float(recall[chosen]), 4),
        false_accepts_per_hour=None if chosen is None else round(float(faph[chosen]), 3),
        curve=[
            {
                "cutoff": round(float(CUTOFFS[i]), 4),
                "recall": round(float(recall[i]), 4),
                "faph": round(float(faph[i]), 3),
            }
            for i in range(0, 256, 5)
        ],
    )
    if chosen is not None:
        report.missed = [
            str(path.relative_to(project.root))
            for path, peak in zip(positives, peaks, strict=True)
            if peak <= CUTOFFS[chosen]
        ]
    project.path("report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report
