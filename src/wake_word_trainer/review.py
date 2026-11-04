"""Collector review: read-only by default; explicit negative policy can change labels."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .datasets import provenance_fields
from .project import Project, ProjectError
from .recordings import SAFE_NAME, read_token

REVIEW_CATEGORIES = {"triggers", "needs_review", "rejected_transcript"}
MAX_AUDIO_BYTES = 4 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
ORIGIN_KEYS = ("category", "device", "filename", "sha256", "created_at", "relative_path")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A Collector token must never follow an HTTP redirect to another host.
        return None


def _get(url: str, token: str):
    request = urllib.request.Request(url, headers={"X-Wakeword-Token": token})
    return urllib.request.build_opener(_NoRedirect()).open(request, timeout=60)


def _read_json(response):
    body = response.read(MAX_JSON_BYTES + 1)
    if len(body) > MAX_JSON_BYTES:
        raise ProjectError("Collector-Prüfantwort ist zu groß")
    return json.loads(body)


def _origin(item: dict) -> dict:
    # Reuse the dataset contract, including its validation. No group IDs are
    # inferred from paths or devices; multiple origins remain separate.
    result = {key: item.get(key) for key in ORIGIN_KEYS} | provenance_fields(item)
    source_hash = item.get("source_upload_sha256")
    if source_hash is not None:
        if not isinstance(source_hash, str) or not re.fullmatch("[a-f0-9]{64}", source_hash):
            raise ProjectError("Ungültige Quellaufnahme-Prüfsumme im Prüfbestand")
        result["source_upload_sha256"] = source_hash
    return result


def fetch(project: Project, output: Path, device: str = "", created_after: str = "", limit: int = 500) -> dict:
    """Snapshot unconfirmed audio; never write into recordings/ or change labels."""
    base = project.config["collector"]["url"].rstrip("/")
    if urllib.parse.urlsplit(base).scheme not in {"http", "https"}:
        raise ProjectError("collector.url muss für die Prüfung gesetzt sein")
    if device and not SAFE_NAME.fullmatch(device):
        raise ProjectError("Ungültiger Gerätename für die Prüfung")
    if not 1 <= limit <= 2000:
        raise ProjectError("Prüflimit muss zwischen 1 und 2000 liegen")
    if created_after:
        try:
            value = datetime.fromisoformat(created_after)
            if value.tzinfo is None:
                raise ValueError("timezone")
        except ValueError as err:
            raise ProjectError("Prüfzeit muss ein ISO-Zeitstempel mit Zeitzone sein") from err
    token = read_token(project)
    prefix = f"/api/wake_word_collector/export/{urllib.parse.quote(project.slug)}/review"
    query = urllib.parse.urlencode({"device": device, "created_after": created_after, "limit": limit})
    try:
        with _get(base + prefix + "?" + query, token) as response:
            listing = _read_json(response)
    except urllib.error.HTTPError as err:
        if err.code == 404:
            raise ProjectError("Collector-Prüfschnittstelle fehlt; Wake Word Collector in HACS aktualisieren") from err
        raise ProjectError(f"Collector-Prüfabruf fehlgeschlagen (HTTP {err.code})") from err
    if not isinstance(listing, dict) or listing.get("labels_confirmed") is not False:
        raise ProjectError("Collector antwortet nicht mit einem unbestätigten Prüfbestand")
    items = listing.get("clips", [])
    if not isinstance(items, list) or len(items) > limit:
        raise ProjectError("Ungültiger Prüfbestand vom Collector")
    locations = set()
    for item in items:
        if not isinstance(item, dict):
            raise ProjectError("Ungültiger Datensatz im Prüfbestand")
        category, node, name = item.get("category", ""), item.get("device", ""), item.get("filename", "")
        if (
            not isinstance(category, str)
            or category not in REVIEW_CATEGORIES
            or not isinstance(node, str)
            or not isinstance(name, str)
            or not SAFE_NAME.fullmatch(node)
            or not SAFE_NAME.fullmatch(name)
            or not name.endswith(".wav")
            or item.get("audio_url") != f"{prefix}/{category}/{node}/{name}"
        ):
            raise ProjectError("Unsicherer oder nicht passender Audioverweis im Prüfbestand")
        _origin(item)
        location = (category, node, name)
        if location in locations:
            raise ProjectError("Mehrdeutiger Audioverweis im Prüfbestand")
        locations.add(location)
    if output.exists():
        raise ProjectError("Prüfausgabe existiert bereits; einen neuen Snapshot-Ordner wählen")
    output.mkdir(parents=True, mode=0o700)
    records = []
    for item in items:
        with _get(base + item["audio_url"], token) as response:
            data = response.read(MAX_AUDIO_BYTES + 1)
        if not 44 <= len(data) <= MAX_AUDIO_BYTES:
            raise ProjectError("Collector liefert eine ungültige Audio-Größe")
        digest = hashlib.sha256(data).hexdigest()
        if item.get("sha256") and item["sha256"] != digest:
            raise ProjectError("Audio-Prüfsumme stimmt nicht mit dem Collector überein")
        relative = Path(item["category"]) / item["device"] / item["filename"]
        path = output / "clips" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o600)
        records.append({**item, "sha256": digest, "relative_path": str(relative)})
    manifest = {
        "source": "collector.review",
        "labels_confirmed": False,
        "labels_changed": False,
        "clips": records,
        "total": listing.get("total", len(records)),
        "truncated": bool(listing.get("truncated")),
    }
    target = output / "snapshot.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    target.chmod(0o600)
    return manifest


def run(
    project: Project,
    output: Path,
    *,
    device: str = "",
    created_after: str = "",
    limit: int = 500,
    stt_model: str = "medium",
    threads: int = 4,
    models: list[Path] | None = None,
    teachers: list[Path] | None = None,
    stt_backend: str = "faster_whisper",
    stt_device: str = "cpu",
    stt_language: str | None = None,
    stt_cache: Path | None = None,
    verification_model: str = "",
    negative_policy: Path | None = None,
) -> dict:
    from .audit import run as audit
    from .consensus import decide, load_policy

    policy = load_policy(negative_policy) if negative_policy else None
    if policy and (stt_backend != "torch_whisper" or not verification_model or not teachers):
        raise ProjectError("Automatische Negativfreigabe benötigt Torch-STT, zweite STT und geprüften Teacher")
    stt_cache = stt_cache or project.path("models", "stt")
    snapshot = fetch(project, output, device, created_after, limit)
    if not snapshot["clips"]:
        result = {
            "unique_clips": 0,
            "findings": {},
            "labels_changed": False,
            "clips": [],
            "source": "collector.review",
            "truncated": snapshot["truncated"],
        }
    else:
        shadow = Project.create(
            output / "analysis-project",
            project.config["wake_word"],
            project.config["language"],
            tts={"phrases": project.config["tts"]["phrases"]},
            evaluation={"sliding_window_size": project.config["evaluation"]["sliding_window_size"]},
        )
        summary = audit(
            shadow,
            [("unlabelled", output / "clips")],
            output / "audit",
            stt_model,
            models or [],
            [],
            threads,
            teachers or [],
            stt_backend=stt_backend,
            stt_device=stt_device,
            stt_language=stt_language,
            stt_cache=stt_cache,
            verification_model=verification_model,
        )
        by_hash = {}
        for item in snapshot["clips"]:
            by_hash.setdefault(item["sha256"], []).append(_origin(item))
        evidence = []
        seen = set()
        for line in (output / "audit" / "clips.jsonl").read_text().splitlines():
            item = json.loads(line)
            digest = item.get("sha256")
            if not isinstance(digest, str) or digest not in by_hash or digest in seen:
                raise ProjectError("Audit-Prüfsumme ist unbekannt oder mehrdeutig")
            seen.add(digest)
            origins = by_hash[digest]
            # Audit deduplicates content, not recording provenance. Never pick
            # one of several source events silently as the target of a POST.
            origin = origins[0] if len(origins) == 1 else {key: None for key in ORIGIN_KEYS}
            evidence.append(
                origin
                | {"sha256": digest, "origins": origins, "origin_ambiguous": len(origins) != 1}
                | {
                    key: item.get(key)
                    for key in ("transcript", "words", "stt", "verification", "findings", "scores", "teachers", "error")
                }
            )
        result = {
            **summary,
            "clips": evidence,
            "source": "collector.review",
            "truncated": snapshot["truncated"],
            "labels_changed": False,
            "output": str(output),
        }
    if policy:
        result["decisions"] = []
        positives = [project.config["wake_word"], *project.config["tts"]["phrases"]]
        for row in result["clips"]:
            decision = decide(row, positives, policy)
            row["decision"] = decision
            if decision["eligible"]:
                prefix = f"/api/wake_word_collector/export/{urllib.parse.quote(project.slug)}/review"
                url = project.config["collector"]["url"].rstrip("/") + prefix
                url += f"/{row['category']}/{row['device']}/{row['filename']}"
                body = {
                    "sha256": row["sha256"],
                    "decision": "negative",
                    "review_id": output.name,
                    "teacher_hashes": decision["teacher_hashes"],
                    "reason": "known_negative_teacher_stt_consensus",
                }
                request = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode(),
                    method="POST",
                    headers={"X-Wakeword-Token": read_token(project), "Content-Type": "application/json"},
                )
                try:
                    with urllib.request.build_opener(_NoRedirect()).open(request, timeout=60) as response:
                        applied = _read_json(response)
                    if (
                        not isinstance(applied, dict)
                        or applied.get("category") != "negatives"
                        or applied.get("sha256") != row["sha256"]
                    ):
                        raise ProjectError("Collector bestätigt Negativentscheidung nicht")
                    row["application"] = {"state": "applied", "category": "negatives"}
                except (urllib.error.URLError, ProjectError, json.JSONDecodeError, UnicodeError, TimeoutError) as err:
                    row["application"] = {"state": "failed", "error": str(err)}
                result["decisions"].append({"sha256": row["sha256"], **row["application"]})
        result["labels_changed"] = sum(d["state"] == "applied" for d in result["decisions"])
        result["application_errors"] = sum(d["state"] == "failed" for d in result["decisions"])
    result.update(created_at=datetime.now(timezone.utc).isoformat(), run_id=output.name)
    target = project.path("review.json")
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    temporary.chmod(0o600)
    temporary.replace(target)
    return result
