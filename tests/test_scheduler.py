"""Night training respects user activity, changed data and cooldowns."""

import json
import sys
from datetime import datetime, time, timezone
from types import SimpleNamespace

import pytest

from wake_word_trainer.project import Project
from wake_word_trainer.scheduler import IdleProbeError, Scheduler, in_window
from wake_word_trainer.state import State


@pytest.mark.parametrize("clock,expected", [("00:30", False), ("01:00", True), ("05:59", True), ("06:00", False)])
def test_window(clock, expected):
    assert in_window(time.fromisoformat(clock), "01:00", "06:00") is expected
    assert in_window(time.fromisoformat("00:30"), "23:00", "05:00")
    assert not in_window(time.fromisoformat("12:00"), "23:00", "05:00")


def setup_scheduler(tmp_path, monkeypatch):
    p = Project.create(tmp_path, "Hey Nova")
    process = SimpleNamespace(poll=lambda: None, returncode=None)
    calls = []
    service = SimpleNamespace(project=p, profiles={"recommended": {}}, process=process, running=lambda: False)
    service.start = lambda profile: calls.append(("start", profile)) or (202, {})
    service.stop = lambda: calls.append(("stop", None))
    scheduler = Scheduler(service, {"enabled": True, "idle_command": ["idle"], "min_new_samples": 2})
    scheduler.saved["baseline"] = ["old"]
    monkeypatch.setattr(scheduler, "idle", lambda: 1800)
    monkeypatch.setattr(scheduler, "samples", lambda: {"old", "new1", "new2"})
    monkeypatch.setattr("wake_word_trainer.service.gpu", lambda: {"use_percent": 0})
    monkeypatch.setattr("wake_word_trainer.scheduler.os.getloadavg", lambda: (0, 0, 0))
    return scheduler, service, calls


def test_auto_training_and_user_return(tmp_path, monkeypatch):
    s, _service, calls = setup_scheduler(tmp_path, monkeypatch)
    at = datetime(2026, 10, 6, 2, tzinfo=timezone.utc)
    s.tick(at)
    assert calls == [("start", "recommended")]
    assert s.status["state"] == "training"
    monkeypatch.setattr(s, "idle", lambda: 3)
    s.tick(at)
    assert calls[-1] == ("stop", None)
    assert s.status["state"] == "interrupted_for_activity"
    monkeypatch.setattr(s, "idle", lambda: 1800)
    s.tick(at)
    assert s.status["state"] == "cooldown"
    assert len(calls) == 2


def test_success_consumes_only_start_snapshot(tmp_path, monkeypatch):
    s, service, _calls = setup_scheduler(tmp_path, monkeypatch)
    at = datetime(2026, 10, 6, 2, tzinfo=timezone.utc)
    s.tick(at)
    State(service.project).update(state="completed")
    service.process.poll = lambda: 0
    service.process.returncode = 0
    monkeypatch.setattr(s, "samples", lambda: {"old", "new1", "new2", "arrived_during_training"})
    s.tick(at)
    assert set(s.saved["baseline"]) == {"old", "new1", "new2"}
    assert "last_completed_at" in json.loads(s.path.read_text())


@pytest.mark.parametrize("reason", ["day", "active", "gpu", "unknown_gpu", "rollout", "few_samples", "manual"])
def test_conditions_block_training(tmp_path, monkeypatch, reason):
    s, service, calls = setup_scheduler(tmp_path, monkeypatch)
    at = datetime(2026, 10, 6, 2, tzinfo=timezone.utc)
    if reason == "day":
        at = at.replace(hour=12)
    if reason == "active":
        monkeypatch.setattr(s, "idle", lambda: 1)
    if reason == "gpu":
        monkeypatch.setattr("wake_word_trainer.service.gpu", lambda: {"use_percent": 30})
    if reason == "unknown_gpu":
        monkeypatch.setattr("wake_word_trainer.service.gpu", lambda: {})
    if reason == "rollout":
        service.project.path("deployment.json").write_text('{"state":"running"}')
    if reason == "few_samples":
        monkeypatch.setattr(s, "samples", lambda: {"old", "one"})
    if reason == "manual":
        service.running = lambda: True
    s.tick(at)
    assert not calls


def test_failed_idle_probe_reports_its_reason_only(tmp_path):
    p = Project.create(tmp_path, "Hey Nova")
    service = SimpleNamespace(project=p, profiles={"recommended": {}})
    script = (
        "import sys; print('secret', file=sys.stderr); "
        "print('idle probe: no desktop display found', file=sys.stderr); sys.exit(1)"
    )
    s = Scheduler(service, {"enabled": True, "idle_command": [sys.executable, "-c", script]})
    with pytest.raises(IdleProbeError, match=r"^idle probe: no desktop display found$"):
        s.idle()
    s.config["idle_command"] = [sys.executable, "-c", "import sys; print('secret', file=sys.stderr); sys.exit(3)"]
    with pytest.raises(IdleProbeError, match="exited with 3"):
        s.idle()


def test_idle_probe_prefers_gnome_and_finds_session_bus(monkeypatch):
    from wake_word_trainer import idle

    monkeypatch.setattr(idle, "manager_environment", lambda: {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1/bus"})
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/gdbus" if name == "gdbus" else None)
    seen = {}

    def run(command, env):
        seen["command"], seen["bus"] = command, env["DBUS_SESSION_BUS_ADDRESS"]
        return "(uint64 1234567,)\n"

    monkeypatch.setattr(idle, "_run", run)
    assert idle.seconds() == 1234.567
    assert seen["bus"] == "unix:path=/run/user/1/bus" and "org.gnome.Mutter.IdleMonitor.GetIdletime" in seen["command"]
    assert idle.mutter({"DBUS_SESSION_BUS_ADDRESS": "x"}) == 1234.567


def test_idle_probe_never_uses_xwayland(monkeypatch, capsys):
    from wake_word_trainer import idle

    monkeypatch.setattr(idle, "environment", lambda: {"XDG_SESSION_TYPE": "wayland", "DISPLAY": ":0"})
    monkeypatch.setattr(idle, "mutter", lambda env: None)
    assert idle.main() == 1
    assert capsys.readouterr().err.startswith("idle probe: Wayland session")
