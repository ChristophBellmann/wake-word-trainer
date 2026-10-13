"""Quellen mit expliziten Rollen übernehmen und Trainingsdaten vorab prüfen."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import audio
from .project import Project, ProjectError

ROLES = {
    "train_positive": "recordings/train/local",
    "eval_positive": "recordings/eval/local",
    "hard_positive": "recordings/hard/local",
    "train_negative": "recordings/negative/train/local",
    "eval_negative": "recordings/negative/eval/local",
    "synthetic_positive": "tts/positive",
    "synthetic_negative": "tts/negative",
    "speech": "corpus/speech",
    "negative_background": "corpus/negative_background",
    "background": "corpus/background",
    "rir": "corpus/rirs",
    "unlabelled": "review/unlabelled",
    # Abgeleitete Positive bleiben im Archiv, damit kein Elternclip in die
    # Bewertung und seine augmentierte Kopie ins Training gelangt.
    "derived_positive": "archive/derived_positive",
}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(source)


def _digest(source) -> str:
    result = hashlib.sha256()
    for chunk in iter(lambda: source.read(1 << 20), b""):
        result.update(chunk)
    return result.hexdigest()


def validate_wav(path: Path) -> None:
    import soundfile

    try:
        info = soundfile.info(path)
    except Exception as err:
        raise ProjectError(f"WAV nicht lesbar: {path}") from err
    if info.samplerate <= 0 or info.channels < 1 or info.frames <= 0:
        raise ProjectError(f"WAV benötigt eine gültige Abtastrate, Kanäle und Audiodaten: {path}")


def conflicts(items: list[dict]) -> list[dict]:
    by_hash = defaultdict(set)
    for item in items:
        by_hash[item["sha256"]].add(item["role"])
    issues = []
    for sha, roles in by_hash.items():
        labelled = roles - {"unlabelled", "derived_positive", "background", "rir"}
        positive = any("positive" in role for role in labelled)
        negative = any("negative" in role or role == "speech" for role in labelled)
        held = any(role.startswith("eval_") for role in labelled)
        train = any(not role.startswith("eval_") for role in labelled)
        if (positive and negative) or (held and train):
            issues.append(
                {
                    "sha256": sha,
                    "roles": sorted(roles),
                    "reason": "label_conflict" if positive and negative else "holdout_overlap",
                }
            )
    return issues


def import_sources(project: Project, manifest: Path, dry_run: bool = False) -> dict:
    """Manifest: {sources: [{folder, role}]}; keine automatische Labelentscheidung."""
    import yaml

    spec = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("sources"), list) or not spec["sources"]:
        raise ProjectError("Importmanifest benötigt eine nicht leere Liste sources")
    items = []
    for source in spec["sources"]:
        role = source.get("role")
        if role not in ROLES:
            raise ProjectError(f"Unbekannte Datenrolle: {role}")
        folder = Path(source["folder"]).expanduser()
        folder = folder if folder.is_absolute() else manifest.parent / folder
        clips = audio.wavs(folder)
        if not folder.is_dir() or not clips:
            raise ProjectError(f"Leere oder fehlende WAV-Quelle: {folder}")
        for path in clips:
            validate_wav(path)
            sha = digest(path)
            items.append(
                {
                    "source": str(path.resolve()),
                    "sha256": sha,
                    "role": role,
                    "target": str(project.path(ROLES[role], sha + ".wav")),
                }
            )
    issues = conflicts(items + inventory(project))
    if issues:
        raise ProjectError(f"Import enthält {len(issues)} Labelkonflikte oder Überschneidungen mit Bewertungsdaten")
    unique = {(item["role"], item["sha256"]) for item in items}
    result = {"files": len(items), "unique_per_role": len(unique), "roles": dict(Counter(role for role, _ in unique))}
    if dry_run:
        return result
    # Alle Quellen wurden geprüft, bevor die erste Aufnahme kopiert wird.
    for item in items:
        target = Path(item["target"])
        if target.exists():
            if digest(target) != item["sha256"]:
                raise ProjectError(f"Importziel verändert: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".part")
        shutil.copy2(item["source"], temporary)
        temporary.replace(target)
    report = project.path("dataset-import.jsonl")
    with report.open("a", encoding="utf-8") as out:
        for item in items:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    report.chmod(0o600)
    for role, section, key in (
        ("speech", "negatives", "speech_folders"),
        ("negative_background", "negatives", "background_folders"),
        ("background", "augmentation", "background_folders"),
        ("rir", "augmentation", "rir_folders"),
    ):
        folder = ROLES[role]
        if role in result["roles"] and folder not in project.config[section][key]:
            project.config[section][key].append(folder)
    project.path("wakeword.yaml").write_text(project.render_config(), encoding="utf-8")
    return result


def inventory(project: Project, rehash: bool = False) -> list[dict]:
    items = []
    cached = {}
    manifest = project.path("dataset-manifest.jsonl")
    if manifest.is_file() and not rehash:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                cached[str(Path(item["path"]).resolve())] = item
            except (ValueError, KeyError):
                continue

    def entry(path, role):
        stat = path.stat()
        identity = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "inode": stat.st_ino,
            "device": stat.st_dev,
        }
        previous = cached.get(str(path.resolve()), {})
        sha = previous.get("sha256") if all(previous.get(k) == v for k, v in identity.items()) else None
        return {"path": str(path), "role": role, "sha256": sha or digest(path), **identity}

    for role, folder in ROLES.items():
        for path in audio.wavs(project.path(folder)):
            items.append(entry(path, role))
    seen = {(item["role"], str(Path(item["path"]).resolve())) for item in items}
    for role, section, key in (
        ("speech", "negatives", "speech_folders"),
        ("negative_background", "negatives", "background_folders"),
        ("background", "augmentation", "background_folders"),
        ("rir", "augmentation", "rir_folders"),
    ):
        for value in project.config[section][key]:
            folder = project.resolve(value)
            clips = audio.wavs(folder)
            if not clips:
                raise ProjectError(f"Leere oder fehlende WAV-Quelle: {folder}")
            for path in clips:
                identity = (role, str(path.resolve()))
                if identity not in seen:
                    items.append(entry(path, role))
                    seen.add(identity)
    # Geräteordner neben local gehören ebenfalls zu den geprüften Aufnahmen.
    for role in ("train_positive", "eval_positive", "hard_positive", "train_negative", "eval_negative"):
        parent = project.path(ROLES[role]).parent
        for path in audio.wavs(parent):
            if path.is_relative_to(parent / "local"):
                continue
            items.append(entry(path, role))
    return items


def preflight(project: Project, rehash: bool = False) -> dict:
    items = inventory(project, rehash=rehash)
    for item in items:
        validate_wav(Path(item["path"]))
    issues = conflicts(items)
    hashes = {str(Path(item["path"]).resolve()): item["sha256"] for item in items}
    provenance = project.path("dataset-import.jsonl")
    if provenance.is_file():
        for line in provenance.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            actual = hashes.get(str(Path(item["target"]).resolve()))
            if actual is not None and actual != item["sha256"]:
                issues.append({"reason": "import_changed", "path": item["target"]})
    active = sorted(
        (item["role"], item["sha256"]) for item in items if item["role"] not in {"unlabelled", "derived_positive"}
    )
    result = {
        "files": len(items),
        "roles": dict(Counter(item["role"] for item in items)),
        "issues": issues,
        "training_fingerprint": hashlib.sha256(json.dumps(active).encode()).hexdigest(),
    }
    manifest = project.path("dataset-manifest.jsonl")
    with manifest.open("w", encoding="utf-8") as out:
        for item in items:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest.chmod(0o600)
    target = project.path("dataset-check.json")
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    target.chmod(0o600)
    if issues:
        raise ProjectError(f"Datenprüfung: {len(issues)} Konflikte oder Training/Bewertung-Überschneidungen; {target}")
    return result


def check_features(roots: list[Path]) -> int:
    """MMAP-Metadaten und Feature-Raum vor dem TensorFlow-Prozess überprüfen."""
    from mmap_ninja.ragged import RaggedMmap

    count = 0
    for root in roots:
        if not root.is_dir():
            continue
        candidates = sorted(
            [marker.parent for marker in root.rglob("type.ninja") if marker.read_text().strip() == "ragged"],
            key=lambda path: len(path.parts),
        )
        leaves = []
        for candidate in candidates:
            # RaggedMmap speichert seine Formen selbst als verschachtelte RaggedMmap.
            if not any(candidate.is_relative_to(parent) for parent in leaves):
                leaves.append(candidate)
        if not leaves:
            raise ProjectError(f"Keine gültigen Microfrontend-MMAPs: {root}")
        for leaf in leaves:
            try:
                data = RaggedMmap(leaf, mode="r")
                first, last = data[0], data[len(data) - 1]
                if any(item.ndim != 2 or item.shape[1] != 40 or not np.isfinite(item).all() for item in (first, last)):
                    raise ValueError("Feature-Raum muss aus 40 Microfrontend-Bins bestehen")
            except Exception as err:
                raise ProjectError(f"Beschädigter oder inkompatibler Microfrontend-MMAP: {leaf}") from err
            count += 1
    return count
