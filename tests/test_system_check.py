"""Training only starts when memory, CPU and VRAM are free enough, and says why not."""

import subprocess
import sys
import threading
import time

import pytest

from wake_word_trainer import system_check
from wake_word_trainer.project import Project, ProjectError
from wake_word_trainer.service import Service
from wake_word_trainer.state import State, read
from wake_word_trainer.system_check import SystemCheck

GIB = 1024**3


@pytest.mark.real_system
def test_memory_and_amd_vram_from_proc_and_sysfs(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 33554432 kB\nMemAvailable: 3145728 kB\nSwapTotal: 4194304 kB\nSwapFree: 1048576 kB\n")
    assert system_check.memory(meminfo) == {"available_memory_gb": 3.0, "swap_used_gb": 3.0}
    assert system_check.memory(tmp_path / "missing") == {}
    for name, total, used in (("card0", 2, 1), ("card1", 12, 5)):
        device = tmp_path / "drm" / name / "device"
        device.mkdir(parents=True)
        (device / "mem_info_vram_total").write_text(str(total * GIB))
        (device / "mem_info_vram_used").write_text(str(used * GIB))
    if not system_check.shutil.which("nvidia-smi"):
        assert system_check.vram(tmp_path / "drm") == {"free_vram_gb": 7.0, "total_vram_gb": 12.0}


def test_problems_per_phase():
    check = SystemCheck({"min_available_memory_gb": 6, "max_load_per_cpu": 0.5, "min_free_vram_gb": 8})
    busy = {"available_memory_gb": 3.0, "load_per_cpu": 0.9, "free_vram_gb": 2.0}
    assert len(check.problems(busy, "start")) == 2  # memory and load; VRAM is checked after pausing
    assert check.problems(busy, "vram") == ["only 2.0 GB VRAM free, need 8 GB"]
    assert check.problems(busy, "running") == ["only 3.0 GB memory available, need 6 GB"]
    assert check.problems({"load_per_cpu": 0.1}, "vram") == ["free VRAM unknown"]
    assert SystemCheck({"enabled": False}).problems(busy, "start") == []
    assert SystemCheck({}).problems({"available_memory_gb": 64.0, "load_per_cpu": 0.1}, "vram") == []


def test_invalid_settings_are_refused():
    with pytest.raises(ProjectError):
        SystemCheck({"min_ram": 4})
    with pytest.raises(ProjectError):
        SystemCheck({"min_available_memory_gb": "a lot"})


def test_vram_wait_gives_freed_memory_time(monkeypatch):
    readings = iter([1.0, 4.0, 10.0])
    monkeypatch.setattr(system_check, "vram", lambda: {"free_vram_gb": next(readings)})
    check = SystemCheck({"min_free_vram_gb": 8, "wait_seconds": 30})
    assert check.after_pause(sleep=lambda _: None) == []


def _service(tmp_path, **config):
    project = Project.create(tmp_path / "p", "Hey Nova")
    token = tmp_path / "token"
    token.write_text("x" * 32)
    return project, Service({"project": project.root, "token_file": token, **config})


def test_start_refused_when_memory_is_short_keeps_last_state(tmp_path, monkeypatch):
    project, service = _service(tmp_path)
    State(project).update(state="completed", best_model_available=True)
    monkeypatch.setattr(system_check, "memory", lambda: {"available_memory_gb": 2.0})
    started = []
    monkeypatch.setattr("wake_word_trainer.service.subprocess.Popen", lambda *a, **k: started.append(a))
    code, answer = service.start("quick")
    assert code == 503 and not started
    assert "only 2.0 GB memory available" in answer["error"] and answer["problems"]
    assert read(project)["state"] == "completed"


def test_start_refused_and_services_restored_when_vram_stays_busy(tmp_path, monkeypatch):
    calls = []

    def systemctl(command, **kwargs):
        calls.append(command[-2:])
        active = "active" if command[-2] == "is-active" else ""
        return subprocess.CompletedProcess(command, 0, stdout=active)

    monkeypatch.setattr("wake_word_trainer.resources.subprocess.run", systemctl)
    monkeypatch.setattr(system_check, "vram", lambda: {"free_vram_gb": 1.0})
    project, service = _service(
        tmp_path, pause_services=["llm.service"], resource_check={"min_free_vram_gb": 8, "wait_seconds": 0}
    )
    code, answer = service.start("quick")
    assert code == 503 and "VRAM" in answer["error"]
    assert ["stop", "llm.service"] in calls and calls[-1] == ["start", "llm.service"]
    assert read(project)["state"] == "failed" and "VRAM" in read(project)["last_error"]


def test_memory_warning_while_running(tmp_path, monkeypatch):
    project, service = _service(tmp_path)
    real = subprocess.Popen
    monkeypatch.setattr(
        "wake_word_trainer.service.subprocess.Popen",
        lambda command, **kwargs: real([sys.executable, "-c", "import time; time.sleep(5)"], **kwargs),
    )
    assert service.start("quick")[0] == 202
    monkeypatch.setattr(system_check, "memory", lambda: {"available_memory_gb": 1.0})
    watcher = threading.Thread(target=service._watch_resources, args=(service.process,), kwargs={"interval": 0.05})
    watcher.start()
    deadline = time.monotonic() + 3
    while not service.resource_warning and time.monotonic() < deadline:
        time.sleep(0.02)
    assert "only 1.0 GB memory available" in service.status()["resources"]["warning"]
    assert "Resource warning" in project.path("service.log").read_text()
    service.stop()
    watcher.join(timeout=5)
    assert service.status()["resources"]["warning"] == ""
