"""Augmented spectrogram features in the layout microWakeWord trains on.

    features/own/training|validation|testing/own_mmap        your recordings
    features/tts/training[/validation|testing]/tts_mmap        synthetic wake word
    features/tts_negative/training/tts_negative_mmap          synthetic look-alikes

Your own held-out recordings (recordings/eval) are the validation and test
set whenever there are any; synthetic speech then only trains.
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import multiprocessing
import os
import random
import shutil
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from . import audio
from .project import Project, ProjectError
from .recordings import LOCAL

STEP_MS = 10
AUGMENTED_SECONDS = 3.2
Log = Callable[[str], None]
_WORKER_AUGMENT = None
_WORKER_MARKER = None


def _initialize_worker(root: str, downloads: str, augment_kind: bool | None) -> None:
    global _WORKER_AUGMENT, _WORKER_MARKER
    faulthandler.enable()
    diagnostics = Path(root) / "feature-workers"
    diagnostics.mkdir(exist_ok=True)
    _WORKER_MARKER = diagnostics / f"{os.getpid()}.json"
    # Vor dem ersten Framework-Import setzen; kein HIP im CPU-Feature-Prozess.
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        os.environ[key] = "-1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    if augment_kind is not None:
        _WORKER_AUGMENT = augmenter(Project.load(root), Path(downloads), augment_kind)


def _worker_features(job: tuple[str, int, int]) -> list[np.ndarray]:
    path, slide, seed = job
    if _WORKER_MARKER is not None:
        _WORKER_MARKER.write_text(json.dumps({"path": path, "seed": seed, "state": "processing"}))
    random.seed(seed)
    np.random.seed(seed)
    result = list(spectrograms([Path(path)], _WORKER_AUGMENT, 1, slide))
    if _WORKER_MARKER is not None:
        _WORKER_MARKER.unlink(missing_ok=True)
    return result


def parallel_spectrograms(project: Project, downloads: Path, clips: list[Path], repeat: int,
                         slide: int, augment_kind: bool | None) -> Iterator[np.ndarray]:
    """Geordnete, begrenzte CPU-Arbeit; unabhängig von Worker-Anzahl deterministisch."""
    workers = int(project.config["training"]["feature_workers"])
    seed = project.config["training"].get("seed")
    seed = 1337 if seed is None else int(seed)
    jobs = (
        (str(path), slide, int.from_bytes(hashlib.sha256(f"{seed}:{iteration}:{path}".encode()).digest()[:4], "big"))
        for iteration in range(repeat) for path in clips
    )
    def create_pool():
        return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_initialize_worker,
                                   initargs=(str(project.root), str(downloads), augment_kind))

    pool, restarts, pending = create_pool(), 0, deque()

    def submit(job):
        try:
            return pool.submit(_worker_features, job)
        except BrokenProcessPool as err:
            future = Future()
            future.set_exception(err)
            return future

    def consume():
        nonlocal pool, restarts, pending
        while True:
            try:
                result = pending[0][1].result()
            except BrokenProcessPool as err:
                pool.shutdown(wait=True, cancel_futures=True)
                if restarts >= 2:
                    raise ProjectError(
                        f"Feature-Prozess wiederholt abgestürzt; Originalclips bleiben erhalten. "
                        f"Diagnose: {project.path('feature-workers')}"
                    ) from err
                restarts += 1
                print(f"Feature-Prozess abgestürzt: offener Block wird vollständig neu verarbeitet ({restarts}/2)",
                      flush=True)
                pool = create_pool()
                pending = deque((job, submit(job)) for job, _ in pending)
            else:
                pending.popleft()
                return result

    try:
        for job in jobs:
            pending.append((job, submit(job)))
            if len(pending) >= workers * 2:
                yield from consume()
        while pending:
            yield from consume()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def augmenter(project: Project, downloads: Path, truncate_randomly: bool = False):
    """truncate_randomly: long negatives contribute a random window instead of their end."""
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
        truncate_randomly=truncate_randomly,
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

    target.parent.mkdir(parents=True, exist_ok=True)
    marker = target.with_name(target.name + ".incomplete")
    marker.write_text("Feature-Bau läuft oder wurde unterbrochen\n", encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)
    count = 0

    def counted():
        nonlocal count
        for item in generator:
            count += 1
            yield item

    RaggedMmap.from_generator(out_dir=str(target), sample_generator=counted(), batch_size=200, verbose=False)
    marker.unlink()
    return count


def _holdout(
    clips: list[Path], share: float = 0.1, hashes: dict | None = None
) -> tuple[list[Path], list[Path], list[Path]]:
    """Deterministic train/validation/test split for synthetic-only projects."""
    from .datasets import digest

    unique = {}
    for path in clips:
        sha = (hashes or {}).get(str(path)) or digest(path)
        unique.setdefault(sha, path)
    clips = [unique[sha] for sha in sorted(unique)]
    random.Random(1337).shuffle(clips)
    if share <= 0:
        return clips, [], []
    held = max(1, int(len(clips) * share)) if len(clips) >= 3 else 0
    return clips[2 * held :], clips[:held], clips[held : 2 * held]


def speech_clips(project: Project, downloads: Path) -> list[Path]:
    """Extra negative speech: own folders plus downloaded presets, evenly thinned to
    negatives.speech_clips so a huge folder does not dominate."""
    settings = project.config["negatives"]
    folders = [project.resolve(folder) for folder in settings["speech_folders"]]
    folders += [downloads / "speech" / preset for preset in settings["speech_presets"]]
    for folder in folders:
        if not folder.is_dir():
            raise ProjectError(f"Speech folder missing: {folder} (download --speech for presets)")
    clips = audio.wavs(*folders)
    limit = int(settings["speech_clips"])
    if len(clips) > limit > 0:
        clips = [clips[int(index * len(clips) / limit)] for index in range(limit)]
    return clips


def build(project: Project, downloads: Path, log: Log = print) -> dict[str, int]:
    from .datasets import preflight

    preflight(project)
    hashes = {
        row["path"]: row["sha256"]
        for row in (json.loads(line) for line in project.path("dataset-manifest.jsonl").read_text().splitlines())
    }
    seed = project.config["training"].get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    recordings = project.recordings
    own_train = audio.wavs(recordings / "train")
    own_eval = audio.wavs(recordings / "eval")
    own_hard = audio.wavs(recordings / "hard")
    negative_train = audio.wavs(recordings / "negative" / "train")
    # Collector clips sit in device folders, local folders under "local".
    satellite_train = [path for path in negative_train if path.parent.name != LOCAL]
    negative_train = [path for path in negative_train if path.parent.name == LOCAL]
    negative_eval = audio.wavs(recordings / "negative" / "eval")
    tts_pos = audio.wavs(project.path("tts", "positive"))
    minimum = float(project.config["tts"]["min_positive_duration_seconds"])
    if minimum:
        kept = [path for path in tts_pos if len(audio.read(path)) / audio.RATE >= minimum]
        log(f"TTS-Dauerfilter: {len(tts_pos) - len(kept)} zu kurze Positive ausgeschlossen")
        tts_pos = kept
    tts_neg = audio.wavs(project.path("tts", "negative"))
    if not (own_train or own_hard or tts_pos):
        raise ProjectError("Nothing to train on: run fetch and/or generate first")

    augments = {False: augmenter(project, downloads), True: augmenter(project, downloads, truncate_randomly=True)}
    repeat = int(project.config["training"]["own_repeat"])
    root = project.features
    counts: dict[str, int] = {}

    def make(name: str, split: str, clips: list[Path], times: int, slide: int, random_window: bool = False) -> None:
        if not clips:
            return
        log(f"Features {name}/{split}: {len(clips)} clips x {times}")
        use_augment = split == "training" and not (
            name in {"own_negative", "satellite_negative", "speech_extra", "background_negative", "tts_negative"}
            and not project.config["augmentation"]["augment_negatives"]
        )
        generator = (
            parallel_spectrograms(project, downloads, clips, times, slide, random_window if use_augment else None)
            if project.config["training"]["feature_workers"] > 1
            else spectrograms(clips, augments[random_window] if use_augment else None, times, slide)
        )
        counts[f"{name}/{split}"] = write_mmap(root / name / split / f"{name}_mmap", generator)

    # Mined false activations (features/mined) belong to earlier rounds and stay.
    if root.exists():
        for child in root.iterdir():
            if child.name != "mined":
                shutil.rmtree(child)
    make("own", "training", own_train, repeat, 10)
    make("own_hard", "training", own_hard, repeat, 10)
    make("own_negative", "training", negative_train, max(1, repeat // 2), 1, random_window=True)
    make("satellite_negative", "training", satellite_train, max(1, repeat // 2), 1, random_window=True)
    make("own_negative", "validation", negative_eval, 1, 1, random_window=True)
    make("own_negative", "testing", negative_eval, 1, 1, random_window=True)
    speech = speech_clips(project, downloads)
    random.Random(1337).shuffle(speech)
    train, validation, test = _holdout(speech, float(project.config["negatives"]["eval_share"]), hashes)
    make("speech_extra", "training", train, 1, 1, random_window=True)
    make("speech_extra", "validation", validation, 1, 1, random_window=True)
    make("speech_extra", "testing", test, 1, 1, random_window=True)
    background_negatives = audio.wavs(*[project.resolve(p) for p in project.config["negatives"]["background_folders"]])
    random.Random(1337).shuffle(background_negatives)
    train, validation, test = _holdout(background_negatives, float(project.config["negatives"]["eval_share"]), hashes)
    make("background_negative", "training", train, 1, 1, random_window=True)
    make("background_negative", "validation", validation, 1, 1, random_window=True)
    make("background_negative", "testing", test, 1, 1, random_window=True)
    if own_eval:
        make("own", "validation", own_eval, 1, 10)
        make("own", "testing", own_eval, 1, 1)
        make("tts", "training", tts_pos, 2, 10)
    else:
        train, validation, test = _holdout(tts_pos, hashes=hashes)
        make("tts", "training", train, 2, 10)
        make("tts", "validation", validation, 1, 10)
        make("tts", "testing", test, 1, 1)
    make("tts_negative", "training", tts_neg, 1, 1)
    return counts
