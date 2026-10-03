"""Augmented spectrogram features in the layout microWakeWord trains on.

    features/own/training|validation|testing/own_mmap        your recordings
    features/tts/training[/validation|testing]/tts_mmap        synthetic wake word
    features/tts_negative/training/tts_negative_mmap          synthetic look-alikes

Your own held-out recordings (recordings/eval) are the validation and test
set whenever there are any; synthetic speech then only trains.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np

from . import audio
from .project import Project, ProjectError

STEP_MS = 10
AUGMENTED_SECONDS = 3.2
Log = Callable[[str], None]


def augmenter(project: Project, downloads: Path):
    from microwakeword.audio.augmentation import Augmentation

    settings = project.config["augmentation"]
    backgrounds = [str(path) for path in sorted((downloads / "background").glob("*")) if path.is_dir()]
    backgrounds += [str(project.resolve(folder)) for folder in settings["background_folders"]]
    impulses = [str(downloads / "rirs")] if (downloads / "rirs").is_dir() else []
    impulses += [str(project.resolve(folder)) for folder in settings["rir_folders"]]
    return Augmentation(
        augmentation_duration_s=AUGMENTED_SECONDS,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.05,
            "PitchShift": 0.1,
            "BandStopFilter": 0.05,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": float(settings["background_probability"]),
            "Gain": 1.0,
            "RIR": float(settings["rir_probability"]),
        },
        impulse_paths=[path for path in impulses if audio.wavs(path)],
        background_paths=[path for path in backgrounds if audio.wavs(path)],
        background_min_snr_db=settings["background_min_snr_db"],
        background_max_snr_db=settings["background_max_snr_db"],
        min_gain_db=-20,
        max_gain_db=0,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )


def spectrograms(clips: Iterable[Path], augment, repeat: int, slide_frames: int) -> Iterator[np.ndarray]:
    """Every clip `repeat` times, augmented differently each time. slide_frames > 1
    yields shifted copies, which mimics the streaming model in training."""
    from microwakeword.audio.audio_utils import generate_features_for_clip

    clips = list(clips)
    for _ in range(repeat):
        for path in clips:
            samples = audio.read(path)
            samples = augment.augment_clip(samples) if augment is not None else samples
            features = generate_features_for_clip(np.asarray(samples, dtype=np.float32), STEP_MS)
            if slide_frames <= 1 or features.shape[0] <= slide_frames:
                yield features
                continue
            length = features.shape[0] - slide_frames + 1
            for offset in range(slide_frames):
                yield features[offset : offset + length]


def write_mmap(target: Path, generator: Iterator[np.ndarray]) -> int:
    from mmap_ninja.ragged import RaggedMmap

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    def counted():
        nonlocal count
        for item in generator:
            count += 1
            yield item

    RaggedMmap.from_generator(out_dir=str(target), sample_generator=counted(), batch_size=200, verbose=False)
    return count


def _holdout(clips: list[Path], share: float = 0.1) -> tuple[list[Path], list[Path], list[Path]]:
    """Deterministic train/validation/test split for synthetic-only projects."""
    held = max(1, int(len(clips) * share)) if len(clips) >= 3 else 0
    return clips[2 * held :], clips[:held], clips[held : 2 * held]


def build(project: Project, downloads: Path, log: Log = print) -> dict[str, int]:
    own_train = audio.wavs(project.recordings / "train")
    own_eval = audio.wavs(project.recordings / "eval")
    tts_pos = audio.wavs(project.path("tts", "positive"))
    tts_neg = audio.wavs(project.path("tts", "negative"))
    if not (own_train or tts_pos):
        raise ProjectError("Nothing to train on: run fetch and/or generate first")

    augment = augmenter(project, downloads)
    repeat = int(project.config["training"]["own_repeat"])
    root = project.features
    counts: dict[str, int] = {}

    def make(name: str, split: str, clips: list[Path], times: int, slide: int) -> None:
        if not clips:
            return
        log(f"Features {name}/{split}: {len(clips)} clips x {times}")
        counts[f"{name}/{split}"] = write_mmap(
            root / name / split / f"{name}_mmap", spectrograms(clips, augment, times, slide)
        )

    if root.exists():
        shutil.rmtree(root)
    make("own", "training", own_train, repeat, 10)
    if own_eval:
        make("own", "validation", own_eval, 1, 10)
        make("own", "testing", own_eval, 1, 1)
        make("tts", "training", tts_pos, 2, 10)
    else:
        train, validation, test = _holdout(tts_pos)
        make("tts", "training", train, 2, 10)
        make("tts", "validation", validation, 1, 10)
        make("tts", "testing", test, 1, 1)
    make("tts_negative", "training", tts_neg, 1, 10)
    return counts
