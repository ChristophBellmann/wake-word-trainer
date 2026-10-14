"""Unconfirmed snapshots remain independent of training labels and rollout."""

import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from wake_word_trainer.cli import main
from wake_word_trainer.project import Project, ProjectError
from wake_word_trainer.review import fetch, run
from wake_word_trainer.scheduler import Scheduler
from wake_word_trainer.service import Service
from wake_word_trainer.state import State, read

PREFIX = "/api/wake_word_collector/export/hey_nova/review"


def setup_collector(tmp_path, server, *, item_changes=None, listing_changes=None):
    body = b"RIFF" + b"0" * 100
    item = {
        "category": "triggers",
        "device": "bath",
        "filename": "test.wav",
        "sha256": hashlib.sha256(body).hexdigest(),
        "audio_url": PREFIX + "/triggers/bath/test.wav",
    }
    item.update(item_changes or {})
    listing = {"labels_confirmed": False, "clips": [item], "total": 1, "truncated": False}
    listing.update(listing_changes or {})

    def protected(headers, data):
        return (200, data) if headers.get("X-Wakeword-Token") == "private-test-token" else (401, b"")

    fake = server(
        {
            PREFIX: lambda headers: protected(headers, json.dumps(listing).encode()),
            PREFIX + "/triggers/bath/test.wav": lambda headers: protected(headers, body),
        }
    )
    token = tmp_path / "collector.token"
    token.write_text("private-test-token")
    project = Project.create(tmp_path / "p", "Hey Nova", collector={"url": fake.url, "token_file": str(token)})
    return project, fake, body


def test_private_readonly_snapshot(tmp_path, server):
    project, fake, body = setup_collector(tmp_path, server)
    output = tmp_path / "snapshot"
    result = fetch(project, output, device="bath", created_after="2026-10-14T00:00:00+02:00", limit=3)
    assert result["labels_changed"] is False
    assert result["labels_confirmed"] is False
    assert (output / "clips/triggers/bath/test.wav").read_bytes() == body
    assert os.stat(output).st_mode & 0o777 == 0o700
    assert os.stat(output / "snapshot.json").st_mode & 0o777 == 0o600
    assert os.stat(output / "clips/triggers/bath/test.wav").st_mode & 0o777 == 0o600
    assert not project.recordings.exists()
    assert all("/review" in request for request in fake.requests)
    with pytest.raises(ProjectError, match="existiert"):
        fetch(project, output)


@pytest.mark.parametrize(
    "change",
    [
        {"category": "negatives"},
        {"category": []},
        {"device": "../secret"},
        {"filename": "../../secret.wav"},
        {"audio_url": "https://other.example/private.wav"},
        {"audio_url": PREFIX + "/triggers/bath/other.wav"},
    ],
)
def test_reject_untrusted_audio_paths_before_creating_snapshot(tmp_path, server, change):
    project, fake, _ = setup_collector(tmp_path, server, item_changes=change)
    with pytest.raises(ProjectError):
        fetch(project, tmp_path / "snapshot")
    assert len(fake.requests) == 1
    assert not (tmp_path / "snapshot").exists()


def test_reject_checksum_mismatch(tmp_path, server):
    project, _, _ = setup_collector(tmp_path, server, item_changes={"sha256": "0" * 64})
    with pytest.raises(ProjectError, match="Prüfsumme"):
        fetch(project, tmp_path / "snapshot")
    assert not list((tmp_path / "snapshot").rglob("*.wav"))


def test_empty_review_and_completion_never_offer_old_model(tmp_path, server):
    project, _, _ = setup_collector(tmp_path, server, listing_changes={"clips": [], "total": 0})
    State(project).update(state="completed", best_model_available=True)
    assert main(["-p", str(project.root), "review"]) == 0
    assert read(project)["best_model_available"] is False
    result = json.loads(project.path("review.json").read_text())
    assert result["unique_clips"] == 0 and result["labels_changed"] is False
    assert not project.recordings.exists()


def test_analysis_maps_evidence_to_collector_without_training_import(tmp_path, server, monkeypatch):
    project, _, _ = setup_collector(tmp_path, server)

    def audit(shadow, sources, output, *args):
        assert shadow.root != project.root
        assert sources[0][0] == "unlabelled"
        output.mkdir()
        snapshot = json.loads((sources[0][1].parent / "snapshot.json").read_text())
        row = {
            "sha256": snapshot["clips"][0]["sha256"],
            "transcript": "synthetic test speech",
            "findings": ["negative_candidate"],
            "scores": [{"peak": 1.0}],
            "teachers": [{"peak": 0.1}],
        }
        (output / "clips.jsonl").write_text(json.dumps(row) + "\n")
        return {"unique_clips": 1, "findings": {"negative_candidate": 1}, "labels_changed": 0}

    monkeypatch.setattr("wake_word_trainer.audit.run", audit)
    result = run(project, tmp_path / "snapshot")
    assert result["clips"][0]["device"] == "bath"
    assert result["clips"][0]["transcript"] == "synthetic test speech"
    assert not project.recordings.exists()


def test_cpu_review_does_not_pause_gpu_services_or_deploy(tmp_path, monkeypatch):
    project = Project.create(
        tmp_path / "p",
        "Hey Nova",
        deployment={"enabled": True, "reference_model": "reference.tflite", "command": ["flash"]},
    )
    token = tmp_path / "service.token"
    token.write_text("s" * 32)
    service = Service({"project": str(project.root), "token_file": str(token)})
    monkeypatch.setattr("wake_word_trainer.gpu.busy", lambda: True)
    monkeypatch.setattr(service.resources, "acquire", lambda: pytest.fail("CPU review paused GPU services"))
    monkeypatch.setattr(service.system, "after_pause", lambda: pytest.fail("CPU review requires free VRAM"))
    launches = []
    real = subprocess.Popen

    def launch(command, **kwargs):
        launches.append(command)
        assert kwargs["env"]["HIP_VISIBLE_DEVICES"] == "-1"
        return real([sys.executable, "-c", "pass"], **kwargs)

    monkeypatch.setattr("wake_word_trainer.service.subprocess.Popen", launch)
    State(project).update(best_model_available=True)
    project.path("deployment.json").write_text(json.dumps({"state": "ready"}))
    assert service.start("review")[0] == 202
    service.watcher.join(timeout=5)
    assert len(launches) == 1 and "review" in launches[0]
    assert read(project)["best_model_available"] is False
    project.path("review.json").write_text(
        json.dumps({"unique_clips": 1, "created_at": "test", "clips": [{"transcript": "private"}]})
    )
    assert "private" not in json.dumps(service.status())
    assert "review" in service.status()["review_profiles"]


@pytest.mark.parametrize("kind", ["review", "benchmark"])
def test_scheduler_rejects_nontraining_profile(tmp_path, kind):
    project = Project.create(tmp_path / "p", "Hey Nova")
    service = SimpleNamespace(project=project, profiles={"special": {"kind": kind}})
    with pytest.raises(ProjectError, match="training profile"):
        Scheduler(service, {"enabled": True, "profile": "special", "idle_command": ["idle"]})


def test_review_token_never_follows_redirect(tmp_path, server):
    import http.server
    import threading

    project, _, _ = setup_collector(tmp_path, server)
    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/unexpected-target")
            self.end_headers()

        def log_message(self, *args):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    project.config["collector"]["url"] = f"http://127.0.0.1:{httpd.server_port}"
    try:
        with pytest.raises(ProjectError, match="HTTP 302"):
            fetch(project, tmp_path / "snapshot")
        assert len(received) == 1 and "/unexpected-target" not in received
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def test_service_review_endpoint_requires_token_and_disables_caching(tmp_path):
    import http.server
    import threading
    import urllib.error
    import urllib.request

    from wake_word_trainer.service import handler

    project = Project.create(tmp_path / "p", "Hey Nova")
    project.path("review.json").write_text(json.dumps({"clips": [{"transcript": "private test"}]}))
    token = tmp_path / "token"
    token.write_text("s" * 32)
    service = Service({"project": str(project.root), "token_file": str(token)})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler(service))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_port}/v1/review"
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url)
        assert error.value.code == 401
        request = urllib.request.Request(url, headers={"Authorization": "Bearer " + "s" * 32})
        with urllib.request.urlopen(request) as response:
            assert "no-store" in response.headers["Cache-Control"]
            assert json.load(response)["clips"][0]["transcript"] == "private test"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)
