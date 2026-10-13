"""Keine automatische Umetikettierung, Inhaltskonflikte und sichere Wiederaufnahme."""

import json
from pathlib import Path

import numpy as np

from wake_word_trainer import audio, audit
from wake_word_trainer.project import Project


def test_conflicting_sources_and_stt_never_change_labels(tmp_path, monkeypatch):
    project = Project.create(tmp_path / "project", "Hey Nova", "de")
    positive = project.recordings / "eval" / "room" / "p.wav"
    negative = project.recordings / "negative" / "train" / "room" / "n.wav"
    audio.write(positive, np.ones(16000) * 0.1)
    negative.parent.mkdir(parents=True)
    negative.write_bytes(positive.read_bytes())
    before = {path: path.read_bytes() for path in (positive, negative)}
    monkeypatch.setattr(audit, "Recognizer", lambda *args: lambda samples: ("Andere Sprache", []))
    output = project.root / "audit"
    result = audit.run(project, [], output)
    assert result["unique_clips"] == 1
    assert result["findings"]["conflicting_labels"] == 1
    assert result["labels_changed"] == 0
    row = json.loads((output / "clips.jsonl").read_text())
    assert row["requires_listening"] and len(row["sources"]) == 2
    assert all(path.read_bytes() == content for path, content in before.items())


def test_failed_recognition_cannot_propose_negative_and_is_retried(tmp_path, monkeypatch):
    project = Project.create(tmp_path / "project", "Hey Nova", "de")
    folder = tmp_path / "triggers"
    audio.write(folder / "unknown.wav", np.ones(16000) * 0.1)

    def fail(*args):
        raise RuntimeError("Testfehler")

    monkeypatch.setattr(audit, "Recognizer", fail)
    output = project.root / "audit"
    assert audit.run(project, [("unlabelled", folder)], output)["findings"] == {"processing_error": 1}
    monkeypatch.setattr(audit, "Recognizer", lambda *args: lambda samples: ("", []))
    assert audit.run(project, [("unlabelled", folder)], output)["findings"] == {"unclear_audio": 1}
    monkeypatch.setattr(audit, "Recognizer", fail)  # unveränderte erfolgreiche Clips aus dem Cache
    assert audit.run(project, [("unlabelled", folder)], output)["findings"] == {"unclear_audio": 1}
    assert len((output / "clips.jsonl").read_text().splitlines()) == 1
    # Neue erlaubte Schreibweisen verlangen eine neue Prüfung.
    assert audit.run(project, [("unlabelled", folder)], output, phrases=["Hei Nova"])["findings"] == {
        "processing_error": 1,
    }


def test_negative_with_wake_word_and_model_disagreement_stays_review_only():
    result = audit.findings({"negative"}, [{"phrase": "hey nova"}], [{"word": "Hey"}], [{"peak": 0.99, "cutoff": 0.9}])
    assert result == ["negative_with_stt_match", "hard_negative"]
    assert audit.findings({"unlabelled"}, [], [{"word": "Hallo"}], []) == ["negative_candidate"]
    assert audit.findings({"unlabelled"}, [], [], []) == ["unclear_audio"]


def test_model_cache_changes_when_weights_change(tmp_path, monkeypatch):
    project = Project.create(tmp_path / "project", "Hey Nova", "de")
    audio.write(project.recordings / "train" / "room" / "p.wav", np.ones(16000) * 0.1)
    model = tmp_path / "model.tflite"
    model.write_bytes(b"first")
    calls = []

    class FakeScorer:
        def __init__(self, path, window):
            self.metadata = {"model": str(path), "sha256": Path(path).read_text(), "cutoff": 0.9}

        def __call__(self, samples):
            calls.append(True)
            return {**self.metadata, "peak": 1.0}

    monkeypatch.setattr(audit, "Scorer", FakeScorer)
    monkeypatch.setattr(audit, "Recognizer", lambda *args: lambda samples: ("", []))
    output = project.root / "audit"
    audit.run(project, [], output, models=[model])
    audit.run(project, [], output, models=[model])
    assert len(calls) == 1
    model.write_bytes(b"second")
    audit.run(project, [], output, models=[model])
    assert len(calls) == 2
