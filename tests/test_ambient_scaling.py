"""Raw HF features and already scaled features must reach inference distinctly."""

import sys
from types import SimpleNamespace

import numpy as np

from wake_word_trainer import evaluate
from wake_word_trainer.project import Project


def test_measure_preserves_raw_ambient_scaling(monkeypatch, tmp_path):
    project = Project.create(tmp_path / "project", "Hey Nova")
    raw = np.full((300, 40), 256, dtype=np.uint16)
    scaled = raw.astype(np.float32) * 0.0390625
    seen = []

    class Model:
        stride = 3

        def __init__(self, path):
            pass

        def predict_spectrogram(self, features):
            seen.append(features.dtype)
            if features.dtype == np.uint16:
                features = features.astype(np.float32) * 0.0390625
            # Both representations encode the same level; premature casting
            # would turn the raw one into a false activation instead.
            return np.full(100, float(features.mean() > 10))

    monkeypatch.setitem(sys.modules, "microwakeword.inference", SimpleNamespace(Model=Model))
    monkeypatch.setitem(
        sys.modules, "microwakeword.audio.audio_utils", SimpleNamespace(generate_features_for_clip=None)
    )
    monkeypatch.setattr(evaluate, "_positives", lambda p: ([], "own"))
    monkeypatch.setattr(evaluate, "_ambient_tracks", lambda downloads, extra: [raw, scaled])
    result = evaluate.measure(project, tmp_path / "model.tflite", tmp_path, 5)
    assert seen == [np.dtype("uint16"), np.dtype("float32")]
    assert result.faph[128] == 0
    assert result.hours > 0
