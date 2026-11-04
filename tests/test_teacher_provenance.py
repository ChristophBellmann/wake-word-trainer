"""Teacher source gates run before device selection, using the shared ledger."""

import contextlib
import json
from pathlib import Path

import pytest

from wake_word_trainer import teacher
from wake_word_trainer.datasets import GROUP_FIELDS, PROVENANCE_FIELDS, digest
from wake_word_trainer.project import Project, ProjectError

from .conftest import chirp, write_wav


def clip(path, seed):
    write_wav(path, chirp(seconds=0.04, seed=seed))
    return path


def record_origin(project, path, role, **fields):
    row = {"source": str(path), "target": str(path.resolve()), "sha256": digest(path), "role": role, **fields}
    with project.path("dataset-import.jsonl").open("a") as out:
        out.write(json.dumps(row) + "\n")


def check_report(project):
    return json.loads(project.path("teacher/dataset-check.json").read_text())


def reject_runtime(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Device, lease or model initialization reached before a rejected source gate")

    for name in ("lease", "torch_device", "_train", "network", "Frontend"):
        monkeypatch.setattr(teacher, name, forbidden)


@pytest.mark.parametrize("include_corpus", [False, True])
@pytest.mark.parametrize("field", GROUP_FIELDS)
@pytest.mark.parametrize("location", ["project", "extra", "mixed"])
def test_group_conflicts_precede_any_runtime_initialization(tmp_path, monkeypatch, include_corpus, field, location):
    project = Project.create(tmp_path / "project", "Hey Nova")
    train_folder = project.recordings / "train/room" if location == "project" else tmp_path / "extra_train"
    eval_folder = tmp_path / "extra_eval" if location == "extra" else project.recordings / "eval/room"
    train = clip(train_folder / "a.wav", 1)
    held = clip(eval_folder / "b.wav", 2)
    assert digest(train) != digest(held)
    record_origin(project, train, "train_positive", **{field: "verified-common-group"})
    record_origin(project, held, "eval_positive", **{field: "verified-common-group"})
    reject_runtime(monkeypatch)
    with pytest.raises(ProjectError, match="Training/Bewertung"):
        teacher.train(
            project,
            device="gpu",
            include_corpus=include_corpus,
            extra_positive=[train_folder] if location != "project" else [],
            extra_eval_positive=[eval_folder] if location == "extra" else [],
        )
    report = check_report(project)
    assert report["provenance"]["status"] == "conflict"
    assert any(issue.get("field") == field and issue["reason"] == "provenance_overlap" for issue in report["issues"])


def test_extra_ledger_binding_is_path_and_hash_and_does_not_modify_sources(tmp_path):
    project = Project.create(tmp_path / "project", "Hey Nova")
    train = clip(tmp_path / "extra_train/a.wav", 3)
    held = clip(tmp_path / "extra_eval/b.wav", 4)
    for index, path in enumerate([train, held]):
        fields = {field: f"{field}-{index}" for field in PROVENANCE_FIELDS}
        fields["capture_generation"] = "same-pipeline"
        record_origin(project, path, "train_positive" if index == 0 else "eval_positive", **fields)
    originals = {
        p: p.read_bytes() for p in [train, held, project.path("dataset-import.jsonl"), project.path("wakeword.yaml")]
    }
    selected = teacher.training_sources(project, extra_positive=[train.parent], extra_eval_positive=[held.parent])
    assert selected["train_positive"] == [train] and selected["eval_positive"] == [held]
    assert check_report(project)["provenance"]["status"] == "known_groups_disjoint"
    rows = [json.loads(line) for line in project.path("teacher/dataset-manifest.jsonl").read_text().splitlines()]
    assert all(row["capture_generation"] == "same-pipeline" for row in rows)
    assert {row["session_id"] for row in rows} == {"session_id-0", "session_id-1"}
    assert {p: p.read_bytes() for p in originals} == originals
    assert not project.path("dataset-manifest.jsonl").exists()
    assert not project.recordings.exists()


def test_unknown_and_partial_extra_provenance_stay_explicit(tmp_path):
    project = Project.create(tmp_path / "project", "Hey Nova")
    train = clip(tmp_path / "extra_train/a.wav", 5)
    held = clip(tmp_path / "extra_eval/b.wav", 6)
    teacher.training_sources(project, extra_positive=[train.parent], extra_eval_positive=[held.parent])
    assert check_report(project)["provenance"]["status"] == "unknown"
    record_origin(project, train, "train_positive", session_id="verified-session")
    teacher.training_sources(project, extra_positive=[train.parent], extra_eval_positive=[held.parent])
    provenance = check_report(project)["provenance"]
    assert provenance["status"] == "incomplete"
    assert provenance["missing"]["session_id"] == 1
    assert provenance["missing"]["speaker_id"] == 2


def test_changed_extra_target_is_rejected_without_reusing_its_old_groups(tmp_path, monkeypatch):
    project = Project.create(tmp_path / "project", "Hey Nova")
    path = clip(tmp_path / "extra/a.wav", 7)
    record_origin(project, path, "train_positive", session_id="old-session")
    clip(path, 8)
    reject_runtime(monkeypatch)
    with pytest.raises(ProjectError, match="veränderte Importe"):
        teacher.train(project, extra_positive=[path.parent])
    report = check_report(project)
    assert {"reason": "import_changed", "path": str(path)} in report["issues"]
    row = json.loads(project.path("teacher/dataset-manifest.jsonl").read_text())
    assert "session_id" not in row


def test_same_hash_at_another_path_does_not_inherit_unproven_groups(tmp_path):
    project = Project.create(tmp_path / "project", "Hey Nova")
    original = clip(tmp_path / "original/a.wav", 9)
    record_origin(project, original, "train_positive", session_id="original-session")
    copy = clip(tmp_path / "copy/a.wav", 9)
    assert digest(original) == digest(copy)
    teacher.training_sources(project, extra_positive=[copy.parent])
    assert check_report(project)["provenance"]["status"] == "unknown"


def test_only_selected_corpus_roles_are_checked(tmp_path):
    project = Project.create(tmp_path / "project", "Hey Nova", tts={"min_positive_duration_seconds": 0})
    held = clip(project.recordings / "eval/room/a.wav", 10)
    tts = clip(project.path("tts/positive/b.wav"), 11)
    record_origin(project, held, "eval_positive", speaker_id="shared-speaker")
    record_origin(project, tts, "synthetic_positive", speaker_id="shared-speaker")
    teacher.training_sources(project, include_corpus=False)
    assert not check_report(project)["issues"]
    with pytest.raises(ProjectError, match="Training/Bewertung"):
        teacher.training_sources(project, include_corpus=True)
    assert check_report(project)["issues"][0]["field"] == "speaker_id"


def test_alias_deduplication_cannot_hide_known_overlap(tmp_path):
    project = Project.create(tmp_path / "project", "Hey Nova")
    clip(project.recordings / "train/room/a.wav", 12)
    alias = clip(tmp_path / "extra/a.wav", 12)
    held = clip(project.recordings / "eval/room/b.wav", 13)
    record_origin(project, alias, "train_positive", parent_recording_id="shared-parent")
    record_origin(project, held, "eval_positive", parent_recording_id="shared-parent")
    with pytest.raises(ProjectError, match="Training/Bewertung"):
        teacher.training_sources(project, extra_positive=[alias.parent])


@pytest.mark.parametrize("empty_exists", [False, True])
def test_missing_or_empty_extra_is_rejected_before_runtime(tmp_path, monkeypatch, empty_exists):
    project = Project.create(tmp_path / "project", "Hey Nova")
    folder = tmp_path / "extra"
    if empty_exists:
        folder.mkdir()
    reject_runtime(monkeypatch)
    with pytest.raises(ProjectError, match="Quelle fehlt oder ist leer"):
        teacher.train(project, extra_eval_positive=[folder])


def test_missing_training_class_fails_before_device_selection(tmp_path, monkeypatch):
    project = Project.create(tmp_path / "project", "Hey Nova")
    clip(project.recordings / "train/room/a.wav", 14)
    reject_runtime(monkeypatch)
    with pytest.raises(ProjectError, match="positive und negative Aufnahmen"):
        teacher.train(project, device="gpu")


def test_invalid_extra_audio_fails_before_device_selection(tmp_path, monkeypatch):
    project = Project.create(tmp_path / "project", "Hey Nova")
    folder = tmp_path / "extra"
    folder.mkdir()
    (folder / "bad.wav").write_bytes(b"not a WAV")
    reject_runtime(monkeypatch)
    with pytest.raises(ProjectError, match="WAV nicht lesbar"):
        teacher.train(project, device="gpu", extra_positive=[folder])


def test_valid_source_selection_is_passed_to_worker_after_check(tmp_path, monkeypatch):
    project = Project.create(tmp_path / "project", "Hey Nova")
    for index, folder in enumerate(["train", "eval", "negative/train", "negative/eval"]):
        clip(project.recordings / folder / "room/a.wav", 20 + index)
    calls = []
    original = teacher.training_sources

    def checked(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append("checked")
        return result

    @contextlib.contextmanager
    def lease(*args):
        assert calls == ["checked"]
        calls.append("lease")
        yield

    def device(*args):
        assert check_report(project)["provenance"]["status"] == "unknown"
        calls.append("device")
        return object()

    def fit(*args, sources, **kwargs):
        assert all(len(paths) == 1 for paths in sources.values())
        assert all(isinstance(path, Path) for paths in sources.values() for path in paths)
        calls.append("fit")
        return project.path("teacher/not-created.pt")

    monkeypatch.setattr(teacher, "training_sources", checked)
    monkeypatch.setattr(teacher, "lease", lease)
    monkeypatch.setattr(teacher, "torch_device", device)
    monkeypatch.setattr(teacher, "_train", fit)
    teacher.train(project, device="gpu")
    assert calls == ["checked", "lease", "device", "fit"]
    assert not project.path("teacher/not-created.pt").exists()
