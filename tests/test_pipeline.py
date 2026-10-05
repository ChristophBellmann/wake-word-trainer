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
    for index in range(6):
        write_wav(tmp_path / "hard" / f"{index}.wav", chirp(seed=100 + index) * 0.2 + noise(0.8, index, 0.02))
    for index in range(20):  # own recordings without the wake word: falling tones
        write_wav(tmp_path / "neg" / f"{index}.wav", descending(200 + index))
    for index in range(30):
        write_wav(tmp_path / "speech" / f"{index}.wav", noise(2.0, 400 + index, 0.05))
    project = tmp_path / "project"
    downloads = tmp_path / "downloads"
    fake_downloads(downloads)
    assert main(["-p", str(project), "init", "Hey Nova"]) == 0
    config = (project / "wakeword.yaml").read_text()
    config += (
        "recordings:\n"
        f"  folders: [{tmp_path / 'mine'}]\n"
        f"  hard_folders: [{tmp_path / 'hard'}]\n"
        f"  negative_folders: [{tmp_path / 'neg'}]\n"
        f"negatives: {{speech_folders: [{tmp_path / 'speech'}], speech_clips: 20}}\n"
        "training: {seed: 42, steps: 600, batch_size: 32, eval_step_interval: 100, "
        "own_repeat: 2, mining_samples: 300}\n"
        "evaluation: {max_false_accepts_per_hour: 5}\n"
    )
    (project / "wakeword.yaml").write_text(config)

    common = ["-p", str(project), "--downloads", str(downloads)]
    assert main([*common, "run", "--rounds", "1"]) == 0

    state = json.loads((project / "state.json").read_text())
    assert state["state"] == "completed" and state["progress_percent"] == 100
    assert state["rounds_total"] == 2 and state["accepted_rounds"] + state["rejected_rounds"] == 2
    assert sorted(p.name for p in (project / "rounds").iterdir()) == [
        "round01.json",
        "round01.tflite",
        "round02.json",
        "round02.tflite",
    ]
    for name in ("own", "own_hard", "own_negative", "speech_extra"):
        assert (project / "features" / name / "training").is_dir(), name

    report = json.loads((project / "report.json").read_text())
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}, indent=2))
    assert report["positive_source"] == "own" and report["positives"] > 5
    assert report["own_negatives"] > 0 and report["own_negatives_triggered"] is not None
    assert report["ambient_hours"] == pytest.approx(4 * 90 / 3600, rel=0.05)
    manifest = json.loads((project / "export" / "hey_nova.json").read_text())
    assert (project / "export" / "hey_nova.tflite").stat().st_size > 10_000
    assert manifest["micro"]["probability_cutoff"] == report["probability_cutoff"]
    # A distinct rising tone against noise is easy: even a short training must find it.
    assert report["recall"] >= 0.5

    # Compare the exported model (with its manifest) and the first round's model.
    assert main([*common, "compare", "--current", "--model", str(project / "rounds" / "round01.tflite")]) == 0
    rows = json.loads((project / "compare.json").read_text())
    assert len(rows) == 2 and "own_cutoff" in rows[0] and "own_cutoff" not in rows[1]
    assert rows[0]["own_cutoff"]["probability_cutoff"] == pytest.approx(report["probability_cutoff"], abs=0.004)

    # A second run from scratch must not trip over the existing model folder. 50 steps
    # learn nothing, so this run must refuse to export and keep the previous model.
    exported = (project / "export" / "hey_nova.tflite").read_bytes()
    assert main([*common, "run", "--rounds", "0", "--steps", "50"]) == 2
    assert sorted(p.name for p in (project / "rounds").iterdir()) == ["round01.json", "round01.tflite"]
    state = json.loads((project / "state.json").read_text())
    assert state["state"] == "failed" and "recognizes none" in state["last_error"]
    assert (project / "export" / "hey_nova.tflite").read_bytes() == exported
