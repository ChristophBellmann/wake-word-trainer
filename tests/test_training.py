"""Native training failures must retain a useful trace and signal name."""

import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from wake_word_trainer import training_worker
from wake_word_trainer.project import Project, ProjectError
from wake_word_trainer.training import _stream, train


def test_training_crash_reports_signal_and_enables_trace(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova")
    (project.features / "own").mkdir(parents=True)
    (project.features / "own_negative").mkdir()

    def crash(command, check):
        assert command[:4] == [sys.executable, "-X", "faulthandler", "-m"]
        return subprocess.CompletedProcess(command, -signal.SIGSEGV)

    with pytest.raises(ProjectError, match=r"Training crashed \(SIGSEGV\); see training.log"):
        train(project, tmp_path / "downloads", runner=crash)


def test_stream_preserves_native_crash_trace(tmp_path):
    log = tmp_path / "training.log"
    script = (
        "import resource, signal; resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); signal.raise_signal(signal.SIGABRT)"
    )
    code = _stream([sys.executable, "-X", "faulthandler", "-c", script], log, None)
    assert code == -signal.SIGABRT
    assert "Fatal Python error: Aborted" in log.read_text()
    assert 'File "<string>", line 1' in log.read_text()


def test_worker_seeds_before_loading_backend_and_preserves_arguments(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "tensorflow",
        SimpleNamespace(keras=SimpleNamespace(utils=SimpleNamespace(set_random_seed=lambda seed: calls.append(seed)))),
    )
    monkeypatch.setattr(sys, "argv", ["worker", "--seed", "42", "--train", "1", "mixednet"])
    monkeypatch.setattr(
        training_worker.runpy, "run_module", lambda name, run_name: calls.append((name, run_name, sys.argv.copy()))
    )
    training_worker.main()
    assert calls == [
        42,
        ("microwakeword.model_train_eval", "__main__", ["microwakeword.model_train_eval", "--train", "1", "mixednet"]),
    ]


@pytest.mark.parametrize("seed", [-1, 2**32, True, "42"])
def test_invalid_seed_is_rejected(tmp_path, seed):
    with pytest.raises(ProjectError, match=r"training.seed"):
        Project.create(tmp_path / "p", "Hey Nova", training={"seed": seed})
