"""The whole pipeline on tiny synthetic data: features, training, evaluation, export.

Needs the train extra (TensorFlow); takes a few minutes on a CPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("microwakeword")

from wake_word_trainer.cli import main
from wake_word_trainer.features import STEP_MS, write_mmap

from .conftest import RATE, chirp, noise, write_wav


def descending(seed: int) -> np.ndarray:
    return chirp(seed=seed)[::-1].copy()


def _features(clips):
    from microwakeword.audio.audio_utils import generate_features_for_clip

    for clip in clips:
        yield generate_features_for_clip(clip, STEP_MS)


def fake_downloads(root):
    """Stand-ins for the microWakeWord negative sets and background audio."""
    negatives = root / "negative_datasets"
    for split, seeds in (("training", range(120)), ("validation", range(200, 220)), ("testing", range(300, 320))):
        clips = [noise(2.0, s, 0.03 + 0.01 * (s % 5)) + 0.5 * np.resize(descending(s), 2 * RATE) for s in seeds]
        write_mmap(negatives / "speech" / split / "speech_mmap", _features(clips))
    for split in ("validation_ambient", "testing_ambient"):
        tracks = [noise(90.0, 1000 + s, 0.02) for s in range(4)]
        write_mmap(negatives / "dinner_party_eval" / split / "ambient_mmap", _features(tracks))
    for s in range(5):
        write_wav(root / "background" / "noise" / f"{s}.wav", noise(5.0, 500 + s, 0.05))


@pytest.mark.slow
def test_train_evaluate_export(tmp_path):
    for index in range(60):
        write_wav(tmp_path / "mine" / f"{index}.wav", np.concatenate([noise(0.2, index, 0.005), chirp(seed=index)]))
    project = tmp_path / "project"
    downloads = tmp_path / "downloads"
    fake_downloads(downloads)
    assert main(["-p", str(project), "init", "Hey Nova"]) == 0
    config = (project / "wakeword.yaml").read_text()
    config += (
        f"recordings:\n  folders: [{tmp_path / 'mine'}]\n"
        "training: {steps: 400, batch_size: 32, eval_step_interval: 100, own_repeat: 2}\n"
        "evaluation: {max_false_accepts_per_hour: 5}\n"
    )
    (project / "wakeword.yaml").write_text(config)

    common = ["-p", str(project), "--downloads", str(downloads)]
    assert main([*common, "run"]) == 0

    report = json.loads((project / "report.json").read_text())
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}, indent=2))
    assert report["positive_source"] == "own" and report["positives"] > 5
    assert report["ambient_hours"] == pytest.approx(4 * 90 / 3600, rel=0.05)
    manifest = json.loads((project / "export" / "hey_nova.json").read_text())
    assert (project / "export" / "hey_nova.tflite").stat().st_size > 10_000
    assert manifest["micro"]["probability_cutoff"] == report["probability_cutoff"]
    # A distinct rising tone against noise is easy: even a short training must find it.
    assert report["recall"] >= 0.5
