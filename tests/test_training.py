"""Native training failures must retain a useful trace and signal name."""

import signal
import subprocess
import sys

import pytest

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
