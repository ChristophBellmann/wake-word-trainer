"""Audit device policy before model initialization; only synthetic mocks."""

import contextlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wake_word_trainer import audit
from wake_word_trainer.project import Project, ProjectError


@pytest.mark.parametrize(
    "backend,device,threads",
    [("faster_whisper", "cpu", 1), ("torch_whisper", "cpu", 2), ("torch_whisper", "gpu", 2)],
)
def test_teacher_receives_explicit_validated_device_and_threads(tmp_path, monkeypatch, backend, device, threads):
    project = Project.create(tmp_path / "project", "Hey Fixture")
    teacher_path = tmp_path / "synthetic-teacher.pt"
    active = False
    events = []

    @contextlib.contextmanager
    def outer_lease(label):
        nonlocal active
        assert device == "gpu", "CPU audit acquired a GPU lease"
        assert label == "STT-Datenprüfung"
        active = True
        events.append("lease-enter")
        try:
            yield None
        finally:
            active = False
            events.append("lease-exit")

    class FakeTeacher:
        def __init__(self, path, *, device, threads):
            assert path == teacher_path
            assert device == expected_device and threads == expected_threads
            assert active == (device == "gpu")
            events.append("teacher-init")
            self.metadata = {"model": "fixture", "device": device, "threads": threads}

        def __call__(self, samples):
            assert active == (expected_device == "gpu")
            events.append("teacher-score")
            return {**self.metadata, "peak": 0.0}

    expected_device, expected_threads = device, threads

    def recognizer(model, language, count):
        assert backend == "faster_whisper" and device == "cpu" and count == threads
        events.append("stt-init")
        return lambda samples: ("", [])

    def torch_recognizer(model, language, count, selected, cache):
        assert backend == "torch_whisper" and selected == device and count == threads
        events.append("stt-init")
        return lambda samples: ("", [])

    monkeypatch.setattr("wake_word_trainer.gpu.lease", outer_lease)
    monkeypatch.setitem(sys.modules, "wake_word_trainer.teacher", SimpleNamespace(Teacher=FakeTeacher))
    monkeypatch.setitem(sys.modules, "wake_word_trainer.stt", SimpleNamespace(TorchRecognizer=torch_recognizer))
    monkeypatch.setattr(audit, "Recognizer", recognizer)
    monkeypatch.setattr(audit.audio, "read", lambda path: np.zeros(16, dtype=np.float32))
    monkeypatch.setattr(
        audit,
        "inventory",
        lambda project, extra: [{"sha256": "a" * 64, "sources": [{"label": "unlabelled", "path": "synthetic.wav"}]}],
    )
    result = audit.run(
        project,
        [],
        tmp_path / "report",
        teachers=[teacher_path],
        stt_backend=backend,
        stt_device=device,
        threads=threads,
    )
    assert result["findings"] == {"unclear_audio": 1}
    assert result["config"]["teachers"][0] == {"model": "fixture", "device": device, "threads": threads}
    assert not active
    expected = ["teacher-init", "stt-init", "teacher-score"]
    assert events == (["lease-enter", *expected, "lease-exit"] if device == "gpu" else expected)


@pytest.mark.parametrize("entry", [audit.run, audit._run])
@pytest.mark.parametrize(
    "settings",
    [
        {"stt_device": "auto"},
        {"stt_device": "cuda"},
        {"stt_backend": "unknown"},
        {"stt_backend": "faster_whisper", "stt_device": "gpu"},
        *({"threads": value} for value in (0, -1, 33, True, 1.5, "2", None)),
    ],
)
def test_invalid_runtime_fails_before_lease_inventory_or_model(tmp_path, monkeypatch, entry, settings):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid runtime reached lease, inventory or model initialization")

    monkeypatch.setattr("wake_word_trainer.gpu.lease", forbidden)
    monkeypatch.setattr(audit, "inventory", forbidden)
    monkeypatch.setattr(audit, "Scorer", forbidden)
    monkeypatch.setattr(audit, "Recognizer", forbidden)
    monkeypatch.setitem(sys.modules, "wake_word_trainer.teacher", SimpleNamespace(Teacher=forbidden))
    monkeypatch.setitem(sys.modules, "wake_word_trainer.stt", SimpleNamespace(TorchRecognizer=forbidden))
    with pytest.raises(ProjectError):
        entry(None, [], tmp_path / "unused", teachers=[tmp_path / "unused.pt"], **settings)
    assert not (tmp_path / "unused").exists()


def test_explicit_gpu_teacher_failure_does_not_retry_with_auto_or_cpu(tmp_path, monkeypatch):
    project = Project.create(tmp_path / "project", "Hey Fixture")
    calls = []
    active = False

    @contextlib.contextmanager
    def outer_lease(label):
        nonlocal active
        active = True
        try:
            yield None
        finally:
            active = False

    def teacher(path, *, device, threads):
        assert active
        calls.append((device, threads))
        raise ProjectError("synthetic GPU unavailable")

    monkeypatch.setattr("wake_word_trainer.gpu.lease", outer_lease)
    monkeypatch.setattr(audit, "inventory", lambda *args: [{"sha256": "b" * 64}])
    monkeypatch.setitem(sys.modules, "wake_word_trainer.teacher", SimpleNamespace(Teacher=teacher))
    with pytest.raises(ProjectError, match="GPU unavailable"):
        audit.run(
            project,
            [],
            tmp_path / "report",
            teachers=[tmp_path / "fixture.pt"],
            stt_backend="torch_whisper",
            stt_device="gpu",
            threads=2,
        )
    assert calls == [("gpu", 2)]
    assert not active
