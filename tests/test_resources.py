"""VRAM services are restored after completion, interruption and partial failure."""

import json
import subprocess
import threading

import pytest

from wake_word_trainer.project import ProjectError
from wake_word_trainer.resources import Resources


def systemd(monkeypatch, active, fail_stop=None):
    calls = []

    def run(command, **kwargs):
        action, unit = command[-2:]
        calls.append((action, unit))
        if action == "is-active":
            return subprocess.CompletedProcess(
                command, 0 if unit in active else 3, stdout="active" if unit in active else "inactive"
            )
        if action == "stop":
            if unit == fail_stop:
                raise subprocess.CalledProcessError(1, command)
            active.discard(unit)
        elif action == "start":
            active.add(unit)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("wake_word_trainer.resources.subprocess.run", run)
    return calls


def test_only_previously_active_services_are_restored(tmp_path, monkeypatch):
    active = {"llm.service"}
    calls = systemd(monkeypatch, active)
    resources = Resources(tmp_path, ["llm.service", "tts.service"])
    resources.acquire()
    assert not active
    assert json.loads(resources.path.read_text()) == ["llm.service"]
    resources.release()
    assert active == {"llm.service"}
    assert ("start", "tts.service") not in calls
    assert not resources.path.exists()


def test_restart_recovers_paused_services(tmp_path, monkeypatch):
    active = {"llm.service", "tts.service"}
    systemd(monkeypatch, active)
    Resources(tmp_path, sorted(active)).acquire()
    assert not active
    Resources(tmp_path, []).release()
    assert active == {"llm.service", "tts.service"}


def test_partial_stop_failure_restores_services(tmp_path, monkeypatch):
    active = {"llm.service", "tts.service"}
    systemd(monkeypatch, active, fail_stop="tts.service")
    with pytest.raises(subprocess.CalledProcessError):
        Resources(tmp_path, ["llm.service", "tts.service"]).acquire()
    assert active == {"llm.service", "tts.service"}
    assert not (tmp_path / "paused_services.json").exists()


@pytest.mark.parametrize("units", ["llm.service", [123], ["--help"], ["../llm.service"]])
def test_invalid_resource_configuration(tmp_path, units):
    with pytest.raises(ProjectError, match="pause_services"):
        Resources(tmp_path, units)


def test_service_restores_resources_after_natural_exit(tmp_path, monkeypatch):
    import sys

    from wake_word_trainer.project import Project
    from wake_word_trainer.service import Service

    active = {"llm.service"}
    systemd(monkeypatch, active)
    project = Project.create(tmp_path / "p", "Hey Nova")
    token = tmp_path / "token"
    token.write_text("x" * 32)
    service = Service({"project": project.root, "token_file": token, "pause_services": ["llm.service"]})
    real = subprocess.Popen
    monkeypatch.setattr(
        "wake_word_trainer.service.subprocess.Popen",
        lambda command, **kwargs: real([sys.executable, "-c", "import time; time.sleep(0.1)"], **kwargs),
    )
    assert service.start("quick")[0] == 202
    assert not active
    assert isinstance(service.watcher, threading.Thread)
    service.watcher.join(timeout=5)
    assert not service.watcher.is_alive()
    assert active == {"llm.service"}


def test_rollout_starts_after_gpu_resources_restore(tmp_path, monkeypatch):
    import sys

    from wake_word_trainer.deployment import record
    from wake_word_trainer.project import Project
    from wake_word_trainer.service import Service
    from wake_word_trainer.state import State

    active = {"llm.service"}
    systemd(monkeypatch, active)
    project = Project.create(
        tmp_path / "p",
        "Hey Nova",
        deployment={"enabled": True, "reference_model": "deployed/model.tflite", "command": ["flash"]},
    )
    token = tmp_path / "token"
    token.write_text("x" * 32)
    service = Service({"project": project.root, "token_file": token, "pause_services": ["llm.service"]})
    service.resources.acquire()
    assert not active
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    service.process = process
    State(project).update(state="completed", best_model_available=True)
    record(project, "ready")
    real = subprocess.Popen
    calls = []

    def launch(command, **kwargs):
        assert active == {"llm.service"}
        calls.append(command)
        return real([sys.executable, "-c", "pass"], **kwargs)

    monkeypatch.setattr("wake_word_trainer.service.subprocess.Popen", launch)
    service._finished(process)
    assert calls[0][2] == "wake_word_trainer.deployment"

    # A data-only run must never deploy an old ready model.
    calls.clear()
    State(project).update(state="completed", best_model_available=False)
    service._finished(process)
    assert not calls


def test_ui_training_blocked_during_cli_rollout(tmp_path, monkeypatch):
    from wake_word_trainer.deployment import record
    from wake_word_trainer.project import Project
    from wake_word_trainer.service import Service

    active = {"llm.service"}
    systemd(monkeypatch, active)
    project = Project.create(tmp_path / "p", "Hey Nova")
    token = tmp_path / "token"
    token.write_text("x" * 32)
    service = Service({"project": project.root, "token_file": token, "pause_services": ["llm.service"]})
    record(project, "running")
    assert service.start("quick")[0] == 409
    assert active == {"llm.service"}


def test_service_restart_keeps_a_live_external_rollout_blocking(tmp_path, monkeypatch):
    import os

    from wake_word_trainer.deployment import record
    from wake_word_trainer.project import Project
    from wake_word_trainer.service import Service

    active = {"llm.service"}
    systemd(monkeypatch, active)
    project = Project.create(tmp_path / "p", "Hey Nova")
    token = tmp_path / "token"
    token.write_text("x" * 32)
    record(project, "running", pid=os.getpid())
    service = Service({"project": project.root, "token_file": token})
    assert service.status()["deployment"]["state"] == "running"
    assert service.start("quick")[0] == 409
