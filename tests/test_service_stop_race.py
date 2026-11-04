"""Intentional stop and early worker failure must keep distinct outcomes."""

import threading

from wake_word_trainer.project import Project
from wake_word_trainer.service import Service
from wake_word_trainer.state import State, read


class Process:
    pid = 424242

    def __init__(self):
        self.returncode = None
        self.finished = threading.Event()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        assert self.finished.wait(3 if timeout is None else timeout)
        return self.returncode


def controller(tmp_path, monkeypatch):
    project = Project.create(tmp_path / "project", "Hey Nova")
    token = tmp_path / "token"
    token.write_text("s" * 32)
    service = Service({"project": str(project.root), "token_file": str(token)})
    monkeypatch.setattr(service.resources, "release", lambda: None)
    service.process = Process()
    State(project).update(state="starting")
    return service, project


def test_stop_records_intent_before_completion_watcher(tmp_path, monkeypatch):
    service, project = controller(tmp_path, monkeypatch)
    stopper = threading.current_thread()
    watcher_done = threading.Event()
    lock = threading.Lock()

    class WatcherFirstAfterUnlock:
        def __enter__(self):
            lock.acquire()

        def __exit__(self, *args):
            lock.release()
            if threading.current_thread() is stopper:
                # Force the watcher to finish at the first unlock in stop().
                assert watcher_done.wait(3)

    service.lock = WatcherFirstAfterUnlock()

    def interrupt(pid, _signal):
        assert pid == service.process.pid
        service.process.returncode = -2
        service.process.finished.set()

    monkeypatch.setattr("wake_word_trainer.service.os.killpg", interrupt)

    def finish():
        try:
            service._finished(service.process)
        finally:
            watcher_done.set()

    watcher = threading.Thread(target=finish)
    watcher.start()
    try:
        assert service.stop() == (200, {"stopped": True})
        assert read(project)["state"] == "stopped"
    finally:
        watcher.join(timeout=3)
    assert not watcher.is_alive()


def test_unexpected_early_exit_remains_failed(tmp_path, monkeypatch):
    service, project = controller(tmp_path, monkeypatch)
    service.process.returncode = 1
    service.process.finished.set()
    service._finished(service.process)
    assert read(project)["state"] == "failed"
