"""Migration erhält Rollen, Bewertungsgrenzen und angenommene Checkpoints."""

import io
import json

import numpy as np
import pytest
import yaml

from wake_word_trainer import audio, downloads, features, pipeline, recordings
from wake_word_trainer.datasets import import_sources, preflight
from wake_word_trainer.evaluate import Report
from wake_word_trainer.project import Project, ProjectError
from wake_word_trainer.training import TFLITE, config, model_args

from .conftest import chirp, noise, write_wav


def spec(tmp_path, sources):
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"sources": sources}))
    return path


def test_import_keeps_explicit_holdout_on_collector_sync(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova", recordings={"folders": [str(tmp_path / "fresh")]})
    write_wav(tmp_path / "held" / "a.wav", chirp(seed=8))
    write_wav(tmp_path / "fresh" / "b.wav", chirp(seed=9))
    manifest = spec(tmp_path, [{"folder": "held", "role": "eval_positive"}])
    assert import_sources(project, manifest, dry_run=True)["files"] == 1
    assert not audio.wavs(project.recordings)
    import_sources(project, manifest)
    held = audio.wavs(project.recordings / "eval")[0]
    body = held.read_bytes()
    recordings.fetch(project)
    assert held.read_bytes() == body
    assert not preflight(project)["issues"]
    assert import_sources(project, manifest)["unique_per_role"] == 1
    assert len(audio.wavs(project.recordings / "eval")) >= 1


@pytest.mark.parametrize("role", ["train_negative", "train_positive", "hard_positive"])
def test_import_refuses_conflict_before_copying(tmp_path, role):
    project = Project.create(tmp_path / "p", "Hey Nova")
    body = write_wav(project.recordings / "eval" / "sat1" / "a.wav", chirp())
    path = tmp_path / "incoming" / "different-name.wav"
    path.parent.mkdir()
    path.write_bytes(body)
    with pytest.raises(ProjectError, match="Labelkonflikte"):
        import_sources(project, spec(tmp_path, [{"folder": "incoming", "role": role}]))
    assert not audio.wavs(project.recordings / "train")
    assert not project.path("dataset-import.jsonl").exists()


def test_unlabelled_and_derived_data_do_not_enter_training(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova")
    write_wav(tmp_path / "old" / "a.wav", chirp())
    import_sources(
        project,
        spec(
            tmp_path,
            [
                {"folder": "old", "role": "unlabelled"},
                {"folder": "old", "role": "derived_positive"},
            ],
        ),
    )
    assert not audio.wavs(project.recordings)
    assert len(audio.wavs(project.path("review"))) == 1
    assert len(audio.wavs(project.path("archive"))) == 1


def test_changed_import_is_detected_even_after_timestamp_restoration(tmp_path):
    import os

    project = Project.create(tmp_path / "p", "Hey Nova")
    write_wav(tmp_path / "old" / "a.wav", chirp(seed=1))
    import_sources(project, spec(tmp_path, [{"folder": "old", "role": "train_positive"}]))
    preflight(project)
    path = audio.wavs(project.recordings)[0]
    original = path.stat()
    write_wav(path, chirp(seed=2))
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(ProjectError, match="Konflikte"):
        preflight(project)
    assert json.loads(project.path("dataset-check.json").read_text())["issues"][0]["reason"] == "import_changed"


def test_extra_speech_holdout_deduplicates_identical_audio_under_different_names(tmp_path):
    from wake_word_trainer.datasets import digest

    for seed in range(10):
        write_wav(tmp_path / f"{seed}.wav", chirp(seed=seed))
    (tmp_path / "copy.wav").write_bytes((tmp_path / "0.wav").read_bytes())
    groups = features._holdout(audio.wavs(tmp_path))
    contents = [{digest(path) for path in group} for group in groups]
    assert [len(group) for group in contents] == [8, 1, 1]
    assert not contents[0] & contents[1] and not contents[0] & contents[2] and not contents[1] & contents[2]


def test_evaluation_audio_is_clean_and_speech_holdouts_are_distinct(tmp_path, monkeypatch):
    project = Project.create(
        tmp_path / "p",
        "Hey Nova",
        augmentation={"augment_negatives": False},
        negatives={"speech_folders": [str(tmp_path / "speech")], "eval_share": 0.1},
    )
    write_wav(project.recordings / "train" / "sat1" / "a.wav", chirp(seed=1))
    write_wav(project.recordings / "eval" / "sat1" / "b.wav", chirp(seed=2))
    write_wav(project.recordings / "negative" / "train" / "local" / "n.wav", noise(1, 4))
    for seed in range(20):
        write_wav(tmp_path / "speech" / f"{seed}.wav", noise(1, 100 + seed))
    positive_augmentation = object()
    monkeypatch.setattr(features, "augmenter", lambda *a, **kw: positive_augmentation)
    captured = {}
    monkeypatch.setattr(features, "spectrograms", lambda clips, aug, repeat, slide: (clips, aug))

    def write(target, inputs):
        captured[target.relative_to(project.features).parts[:2]] = inputs
        return len(inputs[0])

    monkeypatch.setattr(features, "write_mmap", write)
    features.build(project, tmp_path, lambda _: None)
    assert captured[("own", "training")][1] is positive_augmentation
    for key in [("own", "validation"), ("own", "testing"), ("own_negative", "training")]:
        assert captured[key][1] is None
    groups = [set(captured[("speech_extra", split)][0]) for split in ("training", "validation", "testing")]
    assert [len(g) for g in groups] == [16, 2, 2]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]


def test_rejected_round_restores_weights_before_the_next_training(tmp_path, monkeypatch):
    project = Project.create(tmp_path / "p", "Hey Nova", training={"mining_rounds": 2})
    monkeypatch.setattr(pipeline, "build", lambda *a: {})
    monkeypatch.setattr(pipeline.mining, "reset", lambda *a: None)
    monkeypatch.setattr(pipeline.mining, "mine", lambda *a: None)
    monkeypatch.setattr(pipeline, "export", lambda *a: project.path("export", "hey_nova.json"))
    starts = []

    def train(*a, **kw):
        weights = project.model_dir / "best_weights.weights.h5"
        starts.append(weights.read_text() if weights.exists() else None)
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_text(str(len(starts)))
        model = project.model_dir / TFLITE
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_text(str(len(starts)))
        return model

    monkeypatch.setattr(pipeline, "train", train)
    reports = iter([0.9, 0.7, 0.8])
    monkeypatch.setattr(
        pipeline, "evaluate", lambda *a: Report("Hey Nova", "test", 10, "own", 1, 5, 0.5, 0.9, next(reports), 0.1)
    )
    result = pipeline.run(project, tmp_path, log=lambda _: None)
    assert starts == [None, "1", "1"]
    assert result.recall == 0.9
    assert (project.model_dir / "best_weights.weights.h5").read_text() == "1"
    assert (project.model_dir / TFLITE).read_text() == "1"


def test_legacy_model_and_specaugment_parameters_reach_backend(tmp_path):
    project = Project.create(
        tmp_path / "p",
        "Hey Nova",
        training={
            "model_profile": "current_wider",
            "time_mask_max_size": 5,
            "time_mask_count": 2,
            "freq_mask_max_size": 3,
            "freq_mask_count": 1,
        },
    )
    (project.features / "own").mkdir(parents=True)
    (project.features / "own_negative").mkdir()
    settings = config(project, tmp_path)
    assert settings["time_mask_count"] == [2] and settings["freq_mask_max_size"] == [3]
    args = model_args("current_wider")
    assert args[args.index("--pointwise_filters") + 1] == "40,40,40,40"
    assert args[args.index("--first_conv_filters") + 1] == "20"


def test_first_round_without_viable_cutoff_fails_clearly_and_preserves_export(tmp_path, monkeypatch):
    project = Project.create(tmp_path / "p", "Hey Nova")
    exported = project.path("export", "hey_nova.tflite")
    exported.parent.mkdir()
    exported.write_bytes(b"previous-good-model")
    candidate = tmp_path / "candidate.tflite"
    candidate.write_bytes(b"candidate")
    monkeypatch.setattr(pipeline, "build", lambda *a: {})
    monkeypatch.setattr(pipeline.mining, "reset", lambda *a: None)
    monkeypatch.setattr(pipeline, "train", lambda *a, **kw: candidate)
    monkeypatch.setattr(
        pipeline, "evaluate", lambda *a: Report("Hey Nova", "test", 10, "own", 1, 5, 0.5, None, None, None)
    )
    with pytest.raises(ProjectError, match="No cutoff meets"):
        pipeline.run(project, tmp_path, log=lambda _: None)
    assert exported.read_bytes() == b"previous-good-model"


def test_seed_experiments_preserve_dataset_and_have_no_collector_or_rollout(tmp_path):
    from wake_word_trainer.experiments import prepare

    project = Project.create(tmp_path / "p", "Hey Nova")
    write_wav(project.recordings / "train" / "sat1" / "a.wav", chirp(seed=1))
    write_wav(project.recordings / "eval" / "sat1" / "b.wav", chirp(seed=2))
    children = prepare(project, project.path("experiments"), [16, 22], "current_wider")
    original = preflight(project)["training_fingerprint"]
    for child in children:
        assert not child.config["deployment"]["enabled"] and not child.config["collector"]["url"]
        assert child.config["training"]["mining_rounds"] == 0
        assert preflight(child)["training_fingerprint"] == original
    assert children[0].config["training"]["seed"] == 16 and children[1].config["training"]["seed"] == 22
    write_wav(children[0].recordings / "train" / "sat1" / "a.wav", chirp(seed=3))
    assert preflight(project)["training_fingerprint"] == original


def test_mmap_preflight_rejects_incomplete_and_wrong_feature_space(tmp_path):
    pytest.importorskip("mmap_ninja")
    from wake_word_trainer.datasets import check_features
    from wake_word_trainer.features import write_mmap

    root = tmp_path / "source"
    root.mkdir()
    with pytest.raises(ProjectError, match="Keine gültigen"):
        check_features([root])
    write_mmap(root / "training" / "source_mmap", iter([np.ones((80, 40), np.float32)]))
    assert check_features([root]) == 1
    write_mmap(root / "training" / "source_mmap", iter([np.ones((80, 32), np.float32)]))
    with pytest.raises(ProjectError, match="inkompatibler"):
        check_features([root])


def test_transcript_filter_and_larger_speech_download_do_not_duplicate(tmp_path, server):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import soundfile

    def clip(seed):
        output = io.BytesIO()
        soundfile.write(output, noise(1, seed), 16000, format="WAV")
        return output.getvalue()

    table = pa.table(
        {
            "audio": [{"bytes": clip(s)} for s in range(4)],
            "text": ["Hey Nova!", "ordinary words", "more speech", "another sentence"],
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    repo = "facebook/voxpopuli"
    fake = server(
        {
            f"/api/datasets/{repo}/tree/main": json.dumps([{"path": "de/train-000.parquet"}]).encode(),
            f"/datasets/{repo}/resolve/main/de/train-000.parquet": buffer.getvalue(),
        }
    )
    folder = downloads.speech("voxpopuli_de", tmp_path, 1, lambda _: None, fake.url, ["hey nova"])
    assert len(audio.wavs(folder)) == 1
    downloads.speech("voxpopuli_de", tmp_path, 3, lambda _: None, fake.url, ["hey nova"])
    assert len(audio.wavs(folder)) == 3
    assert all(
        not np.array_equal(audio.read(p), audio.read(q))
        for p in audio.wavs(folder)
        for q in audio.wavs(folder)
        if p != q
    )
    with pytest.raises(ValueError, match="transcript filter"):
        downloads.speech("voxpopuli_de", tmp_path, 4, lambda _: None, fake.url, ["other wake phrase"])
