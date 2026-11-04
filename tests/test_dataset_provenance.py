"""Explicit provenance survives imports and guards splits beyond byte identity."""

import json

import pytest
import yaml

from wake_word_trainer import audio
from wake_word_trainer.datasets import GROUP_FIELDS, PROVENANCE_FIELDS, import_sources, preflight
from wake_word_trainer.project import Project, ProjectError

from .conftest import chirp, write_wav


def source(tmp_path, name, role, seed=1, **fields):
    write_wav(tmp_path / name / "clip.wav", chirp(seconds=0.02, seed=seed))
    return {"folder": name, "role": role, **fields}


def manifest(tmp_path, *sources):
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"sources": list(sources)}))
    return path


def metadata(suffix):
    return {field: f"{field}-{suffix}" for field in PROVENANCE_FIELDS}


@pytest.mark.parametrize("field", GROUP_FIELDS)
def test_different_audio_in_same_group_is_refused_before_copy(tmp_path, field):
    project = Project.create(tmp_path / "p", "Hey Nova")
    training = source(tmp_path, "train", "train_positive", **{field: "confirmed-group"})
    held = source(tmp_path, "held", "eval_positive", seed=2, **{field: "confirmed-group"})
    for dry_run in (True, False):
        with pytest.raises(ProjectError, match="Überschneidungen"):
            import_sources(project, manifest(tmp_path, training, held), dry_run=dry_run)
    assert not audio.wavs(project.recordings)
    assert not project.path("dataset-import.jsonl").exists()


@pytest.mark.parametrize("field", GROUP_FIELDS)
def test_existing_import_group_blocks_later_training_import(tmp_path, field):
    project = Project.create(tmp_path / "p", "Hey Nova")
    held = source(tmp_path, "held", "eval_positive", **{field: "held-group"})
    import_sources(project, manifest(tmp_path, held))
    preflight(project)
    ledger = project.path("dataset-import.jsonl").read_bytes()
    training = source(tmp_path, "train", "train_negative", seed=2, **{field: "held-group"})
    with pytest.raises(ProjectError, match="Überschneidungen"):
        import_sources(project, manifest(tmp_path, training))
    assert project.path("dataset-import.jsonl").read_bytes() == ledger
    assert not audio.wavs(project.recordings / "negative")


def test_provenance_roundtrip_and_rehash_preserve_groups_without_resplitting(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova")
    training = source(tmp_path, "train", "train_positive", **metadata("train"))
    held = source(tmp_path, "held", "eval_negative", seed=2, **metadata("held"))
    # A shared capture pipeline is not a shared recording, session or speaker.
    training["capture_generation"] = held["capture_generation"] = "pipeline-v2"
    spec = manifest(tmp_path, training, held)
    assert import_sources(project, spec, dry_run=True)["provenance"]["status"] == "known_groups_disjoint"
    import_sources(project, spec)
    original = {p: p.read_bytes() for p in audio.wavs(project.recordings)}
    first = preflight(project)
    import_sources(project, spec)
    assert preflight(project, rehash=True) == first
    assert {p: p.read_bytes() for p in audio.wavs(project.recordings)} == original
    rows = [json.loads(line) for line in project.path("dataset-manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        expected = training if row["role"] == "train_positive" else held
        assert {field: row[field] for field in PROVENANCE_FIELDS} == {
            field: expected[field] for field in PROVENANCE_FIELDS
        }


@pytest.mark.parametrize("field", GROUP_FIELDS)
def test_preflight_detects_group_added_to_existing_ledger(tmp_path, field):
    project = Project.create(tmp_path / "p", "Hey Nova")
    training = source(tmp_path, "train", "train_positive")
    held = source(tmp_path, "held", "eval_negative", seed=2)
    import_sources(project, manifest(tmp_path, training, held))
    preflight(project)  # Cached file hashes must not cache away new provenance.
    ledger = project.path("dataset-import.jsonl")
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    for row in rows:
        row[field] = "confirmed-common-origin"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ProjectError, match="Überschneidungen"):
        preflight(project)
    report = json.loads(project.path("dataset-check.json").read_text())
    assert report["provenance"]["status"] == "conflict"
    assert report["issues"] == [{
        "reason": "provenance_overlap", "field": field, "group": "confirmed-common-origin",
        "roles": ["eval_negative", "train_positive"],
    }]


def test_legacy_and_partial_provenance_are_not_reported_as_independent(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova")
    write_wav(project.recordings / "train" / "sat1" / "a.wav", chirp(seconds=0.02))
    report = preflight(project)
    assert report["issues"] == []  # Legacy data is usable, independence is unknown.
    assert report["provenance"]["status"] == "unknown"
    assert report["provenance"]["missing"] == dict.fromkeys(PROVENANCE_FIELDS, 1)
    held = source(tmp_path, "held", "eval_positive", seed=2, session_id="confirmed-session")
    import_sources(project, manifest(tmp_path, held))
    report = preflight(project)
    assert report["provenance"]["status"] == "incomplete"
    assert report["provenance"]["missing"]["session_id"] == 1
    assert report["provenance"]["missing"]["speaker_id"] == 2


@pytest.mark.parametrize("value", ["", "  ", 7, ["a", "b"]])
def test_invalid_group_identifier_is_rejected(tmp_path, value):
    project = Project.create(tmp_path / "p", "Hey Nova")
    incoming = source(tmp_path, "incoming", "train_positive", session_id=value)
    with pytest.raises(ProjectError, match="session_id"):
        import_sources(project, manifest(tmp_path, incoming))
    assert not project.path("dataset-import.jsonl").exists()


def test_unknown_null_and_inactive_archived_groups_do_not_create_leakage(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova")
    held = source(tmp_path, "held", "eval_positive", session_id="session", speaker_id=None)
    archived = source(tmp_path, "archived", "derived_positive", seed=2, session_id="session")
    import_sources(project, manifest(tmp_path, held, archived))
    report = preflight(project)
    assert not report["issues"]
    assert report["provenance"]["files"] == 1
    assert report["provenance"]["status"] == "incomplete"


def test_conflicting_reimport_cannot_replace_known_provenance(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova")
    incoming = source(tmp_path, "incoming", "train_positive", session_id="original")
    import_sources(project, manifest(tmp_path, incoming))
    ledger = project.path("dataset-import.jsonl").read_bytes()
    incoming["session_id"] = "replacement"
    with pytest.raises(ProjectError, match="Labelkonflikte"):
        import_sources(project, manifest(tmp_path, incoming))
    assert project.path("dataset-import.jsonl").read_bytes() == ledger


def test_changed_audio_does_not_inherit_old_groups(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova")
    incoming = source(tmp_path, "incoming", "train_positive", **metadata("original"))
    import_sources(project, manifest(tmp_path, incoming))
    preflight(project)
    target = audio.wavs(project.recordings)[0]
    write_wav(target, chirp(seconds=0.02, seed=2))
    with pytest.raises(ProjectError, match="Konflikte"):
        preflight(project)
    row = json.loads(project.path("dataset-manifest.jsonl").read_text())
    assert not set(PROVENANCE_FIELDS) & row.keys()
    report = json.loads(project.path("dataset-check.json").read_text())
    assert {"reason": "import_changed", "path": str(target)} in report["issues"]
