"""Deployment must refuse regressions, changed artifacts and incomplete rollouts."""

import json
import sys
from pathlib import Path

import pytest

from wake_word_trainer.deployment import assess, decision, deploy, digest
from wake_word_trainer.project import Project


def metrics(recall=1, triggered=0, faph=0.1):
    return dict(
        positives=40,
        positive_source="own",
        recall=recall,
        false_accepts_per_hour=faph,
        own_negatives=28,
        own_negatives_triggered=triggered,
    )


@pytest.mark.parametrize(
    "new,old,passed",
    [
        (metrics(), metrics(0.9), True),
        (metrics(triggered=0), metrics(triggered=1), True),
        (metrics(), metrics(), False),
        (metrics(0.8), metrics(0.9), False),
        (metrics(triggered=1), metrics(0.9, triggered=0), False),
        (metrics(faph=0.6), metrics(0.9), False),
        (metrics(triggered=2), metrics(0.9, triggered=3), False),
        ({**metrics(), "positive_source": "synthetic"}, metrics(0.9), False),
    ],
)
def test_decision(new, old, passed):
    assert decision(new, old, {"max_false_accepts_per_hour": 0.5, "max_own_negative_share": 0.05})[0] is passed


def test_assess_and_deploy(tmp_path: Path, monkeypatch):
    marker = tmp_path / "flashed"
    project = Project.create(
        tmp_path / "project",
        "Hey Nova",
        deployment={
            "enabled": True,
            "reference_model": "deployed/model.tflite",
            "command": [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('{{sha256}}')"],
        },
    )
    project.export_dir.mkdir()
    model = project.export_dir / "hey_nova.tflite"
    model.write_bytes(b"new")
    model.with_suffix(".json").write_text("{}")
    reference = project.resolve("deployed/model.tflite")
    reference.parent.mkdir()
    reference.write_bytes(b"old")
    reference.with_suffix(".json").write_text("{}")
    project.path("report.json").write_text("{}")
    # Compare the reference at its deployed threshold, not a newly optimized one.
    monkeypatch.setattr(
        "wake_word_trainer.evaluate.compare",
        lambda *a: [
            {"budget": metrics(), "own_cutoff": metrics()},
            {"budget": metrics(), "own_cutoff": metrics(0.9)},
        ],
    )
    assert assess(project, tmp_path)["passed"]
    parity = json.loads(project.path("report.json").read_text())["parity"]
    assert parity["candidate_sha256"] == digest(model)
    model.write_bytes(b"changed")
    assert not deploy(project)
    assert not marker.exists()
    model.write_bytes(b"new")
    reference.write_bytes(b"changed reference")
    assert not deploy(project)
    reference.write_bytes(b"old")
    project.config["deployment"]["command"] = [sys.executable, "-c", "raise SystemExit(1)"]
    assert not deploy(project)
    assert reference.read_bytes() == b"old"
    project.config["deployment"]["command"] = [str(tmp_path / "missing-command")]
    assert not deploy(project)
    assert json.loads(project.path("deployment.json").read_text())["state"] == "failed"
    project.config["deployment"]["command"] = [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('{{sha256}}')"]
    assert deploy(project)
    assert marker.read_text() == digest(model)
    assert reference.read_bytes() == b"new"
    assert json.loads(project.path("deployment.json").read_text())["state"] == "completed"
