"""Project, recordings, synthetic speech, downloads, evaluation maths, export: no TensorFlow needed."""

from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from wake_word_trainer import audio, downloads, recordings, tts
from wake_word_trainer.cli import main
from wake_word_trainer.evaluate import COOLDOWN_SLICES, Report, choose, false_accepts_per_hour, moving_average
from wake_word_trainer.export import export, manifest
from wake_word_trainer.project import Project, ProjectError, slugify
from wake_word_trainer.training import config as training_config

from .conftest import chirp, noise, write_wav

TOKEN = "t" * 43


# -- Project --------------------------------------------------------------------


def test_slugify():
    assert slugify("Hey Jarvis!") == "hey_jarvis"
    assert slugify("Hallo Wüstenfuchs") == "hallo_wustenfuchs"


def test_init_writes_only_essentials_and_changes(tmp_path):
    assert main(["-p", str(tmp_path), "init", "Hey Nova", "--language", "de", "--collector-url", "http://ha:8123"]) == 0
    data = yaml.safe_load((tmp_path / "wakeword.yaml").read_text())
    assert data["wake_word"] == "Hey Nova" and data["slug"] == "hey_nova" and data["language"] == "de"
    assert data["collector"]["url"] == "http://ha:8123"
    assert data["tts"] == {"phrases": ["hey nova"]}
    assert "training" not in data  # defaults stay out of the file
    project = Project.load(tmp_path)
    assert project.config["training"]["steps"] == 20000
    assert main(["-p", str(tmp_path), "init", "Hey Nova"]) == 2  # exists


def test_validation_errors(tmp_path):
    (tmp_path / "wakeword.yaml").write_text("wake_word: Hey Nova\nrecordings: {eval_share: 1.5}\n")
    with pytest.raises(ProjectError, match="eval_share"):
        Project.load(tmp_path)
    with pytest.raises(ProjectError, match=r"No wakeword\.yaml"):
        Project.load(tmp_path / "missing")


# -- Own recordings -----------------------------------------------------------------


def _routes(clips: dict[tuple[str, str], bytes]) -> dict:
    listing = {
        "phrase": "Hey Nova",
        "candidates": [
            {
                "device": device,
                "filename": name,
                "audio_url": f"/api/wake_word_collector/export/hey_nova/{device}/{name}",
            }
            for device, name in clips
        ]
        + [{"device": "../x", "filename": "evil.wav", "audio_url": "/x"}],
    }

    def guarded(body):
        return lambda headers: (200, body) if headers.get("X-Wakeword-Token") == TOKEN else (401, b"")

    routes = {"/api/wake_word_collector/export/hey_nova": guarded(json.dumps(listing).encode())}
    for (device, name), data in clips.items():
        routes[f"/api/wake_word_collector/export/hey_nova/{device}/{name}"] = guarded(data)
    return routes


def _collector(server, clips: dict[tuple[str, str], bytes]):
    return server(_routes(clips))


def test_fetch_from_collector_and_folder_with_stable_split(tmp_path, server):
    clips = {}
    for index in range(30):
        name = f"ha_sat1_20261003T1200{index:02d}_{index:012x}.wav"
        clips[("sat1", name)] = write_wav(tmp_path / "src" / name, chirp(seed=index))
    fake = _collector(server, clips)
    (tmp_path / "token").write_text(TOKEN + "\n")
    local = tmp_path / "mine"
    write_wav(local / "a.wav", chirp(seed=99))
    project = Project.create(
        tmp_path / "p",
        "Hey Nova",
        collector={"url": fake.url, "token_file": str(tmp_path / "token")},
        recordings={"folders": [str(local)]},
    )

    first = recordings.fetch(project)
    assert first.added == 31 and first.train + first.eval == 31 and 0 < first.eval < 31
    assert not any("evil" in p.name for p in audio.wavs(project.root))
    layout = sorted(p.relative_to(project.recordings).as_posix() for p in audio.wavs(project.recordings))
    assert all(p.startswith(("train/sat1/", "eval/sat1/", "train/local/", "eval/local/")) for p in layout)

    again = recordings.fetch(project)
    assert (again.added, again.removed, again.kept) == (0, 0, 31)
    assert layout == sorted(p.relative_to(project.recordings).as_posix() for p in audio.wavs(project.recordings))

    # Rejected in the collector -> gone here.
    del clips[next(iter(clips))]
    fake.routes.clear()
    fake.routes.update(_routes(clips))
    assert recordings.fetch(project).removed == 1


def test_fetch_rejects_wrong_token(tmp_path, server):
    fake = _collector(server, {})
    (tmp_path / "token").write_text("wrong")
    project = Project.create(
        tmp_path / "p", "Hey Nova", collector={"url": fake.url, "token_file": str(tmp_path / "token")}
    )
    with pytest.raises(Exception, match="401"):
        recordings.fetch(project)


def test_split_is_deterministic():
    assert recordings.split_for("00000000" + "0" * 56, 0.2) == "eval"
    assert recordings.split_for("ffffffff" + "0" * 56, 0.2) == "train"


# -- Synthetic speech -----------------------------------------------------------------


def test_generate_calls_piper_per_phrase_and_skips_existing(tmp_path):
    voice = tmp_path / "v.onnx"
    voice.write_bytes(b"x")
    Path(f"{voice}.json").write_text("{}")
    project = Project.create(
        tmp_path / "p",
        "Hey Nova",
        tts={
            "voices": [str(voice)],
            "phrases": ["hey nova", "hey noova"],
            "samples": 5,
            "negative_phrases": ["hey nora"],
            "negative_samples": 2,
        },
    )
    calls = []

    def fake_piper(command, check):
        calls.append(command)
        out = Path(command[command.index("--output-dir") + 1])
        for index in range(int(command[command.index("--max-samples") + 1])):
            write_wav(out / f"{index}.wav", chirp(0.5, index))
        return subprocess.CompletedProcess(command, 0)

    assert tts.generate(project, runner=fake_piper) == (5, 2)
    assert [c[3] for c in calls] == ["hey nova", "hey noova", "hey nora"]
    assert [c[c.index("--max-samples") + 1] for c in calls] == ["3", "2", "2"]
    calls.clear()
    assert tts.generate(project, runner=fake_piper) == (5, 2)
    assert calls == []


def test_generate_needs_voices(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova")
    with pytest.raises(ProjectError, match=r"tts\.voices"):
        tts.generate(project)


# -- Downloads ------------------------------------------------------------------------


def _zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipped:
        for name, data in files.items():
            zipped.writestr(name, data)
    return buffer.getvalue()


def test_download_negative_features_rirs_and_voice(tmp_path, server):
    routes = {}
    for name in downloads.NEGATIVE_SETS:
        routes[f"/datasets/{downloads.NEGATIVE_REPO}/resolve/main/{name}.zip"] = _zip({f"{name}/training/x": b"1"})
    routes[f"/api/datasets/{downloads.RIR_REPO}/tree/main/16khz"] = json.dumps(
        [{"path": "16khz/a.wav"}, {"path": "16khz/readme.txt"}]
    ).encode()
    routes[f"/datasets/{downloads.RIR_REPO}/resolve/main/16khz/a.wav"] = b"RIFF"
    routes["/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"] = b"onnx"
    routes["/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json"] = b"{}"
    fake = server(routes)
    log = []

    target = downloads.negative_features(tmp_path, log.append, fake.url)
    assert sorted(p.name for p in target.iterdir()) == sorted(downloads.NEGATIVE_SETS)
    assert (target / "speech" / "training" / "x").read_bytes() == b"1"
    assert not (tmp_path / "archives" / "speech.zip").exists()
    rirs = downloads.room_impulses(tmp_path, log.append, fake.url)
    assert [p.name for p in rirs.glob("*.wav")] == ["a.wav"]
    count = len(fake.requests)
    downloads.negative_features(tmp_path, log.append, fake.url)
    downloads.room_impulses(tmp_path, log.append, fake.url)
    assert len(fake.requests) == count  # nothing downloaded twice
    onnx = downloads.voice("de_DE-thorsten-medium", tmp_path / "voices", log.append, fake.url)
    assert onnx.read_bytes() == b"onnx" and Path(f"{onnx}.json").is_file()
    with pytest.raises(ValueError):
        downloads.voice("thorsten", tmp_path, log.append, fake.url)


def test_zip_slip_is_refused(tmp_path):
    with zipfile.ZipFile(io.BytesIO(_zip({"../evil": b"x"}))) as zipped, pytest.raises(ValueError):
        downloads._safe_extract_zip(zipped, tmp_path / "t")


# -- Evaluation maths -----------------------------------------------------------------


def test_moving_average():
    assert np.allclose(moving_average(np.array([0, 1, 1, 1, 0]), 3), [2 / 3, 1, 2 / 3])
    assert len(moving_average(np.array([1.0]), 3)) == 0


def test_false_accepts_respect_cooldown():
    track = np.zeros(4000)
    track[100:110] = 0.9  # one activation despite ten slices above the cutoff
    track[100 + COOLDOWN_SLICES + 5] = 0.6  # a second, weaker one after the cooldown
    cutoffs = np.array([0.5, 0.7, 0.95])
    faph, hours = false_accepts_per_hour([track], cutoffs)
    assert hours == pytest.approx(4000 * 0.03 / 3600)
    assert faph * hours == pytest.approx([2, 1, 0])


def test_choose_lowest_cutoff_within_budget():
    faph = np.array([50.0, 3.0, 0.4, 0.1, 0.0])
    assert choose(faph, 0.5) == 2
    assert choose(faph, 0.0) == 4
    assert choose(np.array([5.0, 2.0]), 0.5) is None
    faph = np.zeros(256)
    assert choose(faph, 0.5) == 0
    assert choose(faph, 0.5, minimum=0.5) == 128  # 128/255 is the first step >= 0.5


# -- Training config and export -------------------------------------------------------


def test_training_config_uses_available_sets(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova")
    with pytest.raises(ProjectError, match="positive"):
        training_config(project, tmp_path / "dl")
    (project.features / "own" / "training").mkdir(parents=True)
    with pytest.raises(ProjectError, match="negative"):
        training_config(project, tmp_path / "dl")
    for name in ("speech", "dinner_party_eval"):
        (tmp_path / "dl" / "negative_datasets" / name).mkdir(parents=True)
    config = training_config(project, tmp_path / "dl")
    sets = {Path(item["features_dir"]).name: item for item in config["features"]}
    assert set(sets) == {"own", "speech", "dinner_party_eval"}
    assert sets["own"]["truth"] and sets["own"]["sampling_weight"] == 3.0
    assert sets["dinner_party_eval"]["sampling_weight"] == 0.0
    assert config["training_steps"] == [20000] and config["window_step_ms"] == 10


def test_export_manifest(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova", language="de")
    report = Report("Hey Nova", "m", 40, "own", 2.0, 5, 0.5, 0.8627, 0.95, 0.3)
    model = tmp_path / "m.tflite"
    model.write_bytes(b"TFL3")
    path = export(project, model, report)
    data = json.loads(path.read_text())
    assert data["model"] == "hey_nova.tflite" and (path.parent / "hey_nova.tflite").read_bytes() == b"TFL3"
    assert data["trained_languages"] == ["de"] and data["version"] == 2
    assert data["micro"] == {
        "probability_cutoff": 0.8627,
        "sliding_window_size": 5,
        "feature_step_size": 10,
        "tensor_arena_size": 30000,
        "minimum_esphome_version": "2024.7.0",
    }
    report.probability_cutoff = None
    with pytest.raises(ProjectError):
        manifest(project, report)


def test_audio_roundtrip_resamples(tmp_path):
    from scipy.io import wavfile

    stereo = (np.stack([noise(22050 / 16000, 1), noise(22050 / 16000, 2)], axis=1) * 32767).astype(np.int16)
    wavfile.write(tmp_path / "s.wav", 22050, stereo)
    samples = audio.read(tmp_path / "s.wav")
    assert samples.dtype == np.float32 and len(samples) == 16000
