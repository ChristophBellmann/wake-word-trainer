"""Hard negative mining: what the trained model wrongly reacts to becomes training data.

After a training, the streaming model runs over the negative *training* sets
(never the evaluation sets). Every spectrogram whose averaged score exceeds
the mining threshold is stored in features/mined/training; the next round
trains with it. Each round is evaluated; the best round is kept.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from .evaluate import moving_average
from .features import write_mmap
from .project import Project

Log = Callable[[str], None]
NEGATIVE_SETS = ("speech", "dinner_party", "no_speech")
OWN_NEGATIVE_SETS = ("own_negative", "satellite_negative", "speech_extra", "tts_negative")


def sources(project: Project, downloads: Path) -> list[Path]:
    roots = [downloads / "negative_datasets" / name / "training" for name in NEGATIVE_SETS]
    roots += [project.features / name / "training" for name in OWN_NEGATIVE_SETS]
    return [mmap for root in roots if root.is_dir() for mmap in sorted(root.glob("*_mmap"))]


def _sample(mmaps: list[Path], limit: int) -> Iterator[np.ndarray]:
    """Up to `limit` spectrograms, evenly spread over all sources."""
    from mmap_ninja.ragged import RaggedMmap

    opened = [RaggedMmap(str(path)) for path in mmaps]
    total = sum(len(item) for item in opened)
    if total == 0:
        return
    step = max(1.0, total / limit)
    for item in opened:
        share = max(1, round(len(item) / step))
        for index in np.linspace(0, len(item) - 1, share).astype(int):
            yield np.asarray(item[int(index)])


def mine(project: Project, model_path: Path, downloads: Path, round_number: int, log: Log = print) -> tuple[int, int]:
    """Returns (spectrograms checked, false activations stored)."""
    from microwakeword.inference import Model

    settings = project.config["training"]
    threshold = float(settings["mining_threshold"])
    window = int(project.config["evaluation"]["sliding_window_size"])
    model = Model(str(model_path))
    checked = 0
    found: list[np.ndarray] = []
    for spectrogram in _sample(sources(project, downloads), int(settings["mining_samples"])):
        checked += 1
        averaged = moving_average(np.asarray(model.predict_spectrogram(spectrogram)), window)
        if len(averaged) and averaged.max() > threshold:
            found.append(spectrogram)
    log(f"Mining round {round_number}: {len(found)} of {checked} negative spectrograms activate the model")
    if found:
        write_mmap(project.features / "mined" / "training" / f"round{round_number:02d}_mmap", iter(found))
    return checked, len(found)


def reset(project: Project) -> None:
    shutil.rmtree(project.features / "mined", ignore_errors=True)
