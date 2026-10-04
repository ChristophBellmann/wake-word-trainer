"""Hard and negative recordings, speech presets, progress state, the HTTP service."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import urllib.error
import urllib.request

import pytest
import yaml

from wake_word_trainer import audio, downloads, recordings
from wake_word_trainer.project import Project
from wake_word_trainer.service import make_server
from wake_word_trainer.state import State, read

from .conftest import chirp, noise, write_wav

TOKEN = "t" * 43


def test_fetch_hard_and_negative_clips(tmp_path, server):
    neg = write_wav(tmp_path / "n.wav", noise(1.0, 3))
    listing = {"negatives": [{"device": "sat1", "filename": "ha_sat1_1.wav", "audio_url": "/neg/1"}]}
    fake = server(
        {
            "/api/wake_word_collector/export/hey_nova": json.dumps({"candidates": []}).encode(),
            "/api/wake_word_collector/export/hey_nova/negatives": json.dumps(listing).encode(),
            "/neg/1": neg,
        }
    )
    (tmp_path / "token").write_text(TOKEN)
    for index in range(12):
        write_wav(tmp_path / "pos" / f"{index}.wav", chirp(seed=index))
    for index in range(3):
        write_wav(tmp_path / "hard" / f"{index}.wav", chirp(seed=50 + index))
    for index in range(10):
        write_wav(tmp_path / "neg" / f"{index}.wav", noise(1.0, 60 + index))
    project = Project.create(
        tmp_path / "p",
        "Hey Nova",
        collector={"url": fake.url, "token_file": str(tmp_path / "token")},
        recordings={
            "folders": [str(tmp_path / "pos")],
            "hard_folders": [str(tmp_path / "hard")],
            "negative_folders": [str(tmp_path / "neg")],
        },
    )
    result = recordings.fetch(project)
    assert (result.train + result.eval, result.hard, result.negative) == (12, 3, 11)
    assert all(p.parent.name == "local" for p in audio.wavs(project.recordings / "hard"))
    negatives = [p.relative_to(project.recordings).parts[:2] for p in audio.wavs(project.recordings / "negative")]
    assert {part[1] for part in negatives} <= {"train", "eval"}
    assert any(p.name == "ha_sat1_1.wav" for p in audio.wavs(project.recordings / "negative"))
    assert "difficult" in str(result) and "without the wake word" in str(result)


def test_older_collector_without_negative_export(tmp_path, server):
    pos = write_wav(tmp_path / "p.wav", chirp())
    listing = {"candidates": [{"device": "sat1", "filename": "ha_sat1_1.wav", "audio_url": "/p/1"}]}
    fake = server({"/api/wake_word_collector/export/hey_nova": json.dumps(listing).encode(), "/p/1": pos})
    (tmp_path / "token").write_text(TOKEN)
    project = Project.create(
        tmp_path / "p", "Hey Nova", collector={"url": fake.url, "token_file": str(tmp_path / "token")}
    )
    assert recordings.fetch(project).negative == 0


def test_held_out_recording_cannot_reenter_training_as_difficult_example(tmp_path):
    held = None
    for seed in range(100):
        body = write_wav(tmp_path / "positive" / f"{seed}.wav", chirp(seed=seed))
        if recordings.split_for(hashlib.sha256(body).hexdigest(), 0.2) == "eval":
            held = body
            break
    assert held is not None
    (tmp_path / "hard").mkdir()
    (tmp_path / "hard" / "different_name.wav").write_bytes(held)
    write_wav(tmp_path / "hard" / "new_difficult.wav", chirp(seed=500))
    project = Project.create(
        tmp_path / "p",
        "Hey Nova",
        recordings={"folders": [str(tmp_path / "positive")], "hard_folders": [str(tmp_path / "hard")]},
    )
    # Re-fetch must remove an already imported leaking copy as well.
    stale = project.recordings / "hard" / "local" / "old.wav"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(held)
    result = recordings.fetch(project)
    assert result.eval == 1 and result.hard == 1 and result.removed == 1
    held_hashes = {hashlib.sha256(p.read_bytes()).digest() for p in audio.wavs(project.recordings / "eval")}
    training_hashes = {
        hashlib.sha256(p.read_bytes()).digest()
        for p in audio.wavs(project.recordings / "train", project.recordings / "hard")
    }
    assert not held_hashes & training_hashes
    assert (tmp_path / "hard" / "different_name.wav").read_bytes() == held
    assert recordings.fetch(project).removed == 0


def test_speech_preset_from_parquet(tmp_path, server):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    sf = pytest.importorskip("soundfile")

    def flac(seed: int) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, noise(1.5, seed, 0.1), 24000, format="FLAC")  # 24000 samples = 1 s
        return buffer.getvalue()

    def parquet(seeds) -> bytes:
        table = pa.table(
            {"audio": [{"bytes": flac(s), "path": f"{s}.flac"} for s in seeds], "text": ["x"] * len(seeds)}
        )
        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        return buffer.getvalue()

    repo = "facebook/multilingual_librispeech"
    tree = [
        {"type": "file", "path": "german/train/0.parquet"},
        {"type": "file", "path": "german/train/1.parquet"},
        {"type": "file", "path": "german/test/0.parquet"},
        {"type": "file", "path": "french/train/0.parquet"},
    ]
    fake = server(
        {
            f"/api/datasets/{repo}/tree/main": json.dumps(tree).encode(),
            f"/datasets/{repo}/resolve/main/german/train/0.parquet": parquet(range(3)),
            f"/datasets/{repo}/resolve/main/german/train/1.parquet": parquet(range(3, 6)),
        }
    )
    target = downloads.speech("mls_de", tmp_path, clips=4, log=lambda _: None, base=fake.url)
    wavs = audio.wavs(target)
    assert len(wavs) == 4 and len(audio.read(wavs[0])) == 16000
    assert not any("test" in r or "french" in r for r in fake.requests)
    with pytest.raises(ValueError, match="Unknown speech preset"):
        downloads.speech("klingon", tmp_path, 1)


def test_progress_percent(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova")
    state = State(project)
    state.update(state="running", phase="features", rounds_total=2, round_current=1, steps_total=100)
    assert read(project)["progress_percent"] == 6.0
    state.update(phase="train", step_current=50)
    assert read(project)["progress_percent"] == pytest.approx(10 + 85 * 0.25, abs=0.1)
    state.update(round_current=2, step_current=100, phase="evaluate")
    assert read(project)["progress_percent"] == 95.0
    state.update(state="completed")
    assert read(project)["progress_percent"] == 100


# -- Service ------------------------------------------------------------------------


def _call(base: str, path: str, body: dict | None = None, token: str = TOKEN):
    request = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        method="GET" if body is None else "POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


@pytest.fixture
def service(tmp_path):
    import threading

    project = Project.create(tmp_path / "p", "Hey Nova")
    for index in range(3):
        write_wav(project.recordings / "eval" / "sat1" / f"{index}.wav", chirp(seed=index))
    (tmp_path / "token").write_text(TOKEN)
    played = tmp_path / "played.wav"
    config = {
        "project": str(project.root),
        "token_file": str(tmp_path / "token"),
        "bind": "127.0.0.1",
        "port": 0,
        "profiles": {
            "slow": {"label": "Slow", "rounds": 0},
            "validate": {"label": "Check", "prepare_only": True},
        },
        "speaker_test": {"routes": {"copy": ["cp", "{file}", str(played)], "broken": ["false", "{file}"]}},
    }
    server, svc = make_server(config)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", svc, project, played
    if svc.running():
        svc.process.kill()
    server.shutdown()


def test_service_auth_status_and_files(service):
    base, _, project, _ = service
    assert _call(base, "/health", token="")[0] == 200
    assert _call(base, "/v1/status", token="wrong")[0] == 401
    status, body = _call(base, "/v1/status")
    data = json.loads(body)
    assert status == 200 and data["state"] == "idle" and data["slug"] == "hey_nova"
    assert data["profiles"] == {"slow": "Slow", "validate": "Check"} and data["workstation_online"] is True
    assert _call(base, "/v1/model/hey_nova.json")[0] == 404
    project.export_dir.mkdir(parents=True)
    (project.export_dir / "hey_nova.json").write_text('{"type": "micro"}')
    assert _call(base, "/v1/model/hey_nova.json") == (200, b'{"type": "micro"}')
    assert _call(base, "/v1/model/other.json")[0] == 404
    assert _call(base, "/v1/model/..%2Fwakeword.yaml")[0] == 404


def test_service_speaker_test(service):
    base, _, _, played = service
    status, body = _call(base, "/v1/speaker_test", {"route": "copy"})
    assert status == 200 and json.loads(body)["clip"].startswith("recordings/eval/sat1/")
    assert played.is_file()
    assert _call(base, "/v1/speaker_test", {"route": "nope"})[0] == 400
    assert _call(base, "/v1/speaker_test", {"route": "broken"})[0] == 500


def test_service_start_stop(service, monkeypatch):
    base, _svc, project, _ = service
    commands = []
    real = subprocess.Popen

    def fake_popen(command, **kwargs):
        commands.append(command)
        return real([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)

    monkeypatch.setattr("wake_word_trainer.service.subprocess.Popen", fake_popen)
    assert _call(base, "/v1/start", {"profile": "unknown"})[0] == 400
    assert _call(base, "/v1/start", {"profile": "validate"})[0] == 202
    assert commands[0][-2:] == ["validate", "--prepare-only"] and "--profile" in commands[0]
    assert _call(base, "/v1/start", {"profile": "slow"})[0] == 409
    assert json.loads(_call(base, "/v1/status")[1])["state"] == "starting"
    status, body = _call(base, "/v1/stop", {})
    assert status == 200 and json.loads(body)["stopped"] is True
    assert read(project)["state"] == "stopped"
    assert _call(base, "/v1/start", {"profile": "slow"})[0] == 202
    assert commands[1][-2:] == ["--rounds", "0"]


def test_service_marks_dead_run_failed(service):
    base, _svc, project, _ = service
    State(project).update(state="running", phase="train")
    data = json.loads(_call(base, "/v1/status")[1])
    assert data["state"] == "failed" and "ended unexpectedly" in data["last_error"]


def test_service_config_example_parses():
    from wake_word_trainer import service as module

    example = module.__doc__.split("Configuration (YAML):")[1]
    config = yaml.safe_load("\n".join(line[4:] for line in example.splitlines()))
    assert set(config["profiles"]) == {"quick", "recommended", "thorough", "validate"}
    assert config["speaker_test"]["routes"]["usb"][-1] == "{file}"


def test_service_rejects_short_token(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova")
    (tmp_path / "token").write_text("short")
    with pytest.raises(SystemExit):
        make_server({"project": str(project.root), "token_file": str(tmp_path / "token"), "port": 0})
