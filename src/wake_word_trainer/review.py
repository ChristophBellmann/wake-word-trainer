"""Read-only Collector review snapshots and independent STT/model analysis."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from .project import Project, ProjectError
from .recordings import SAFE_NAME, read_token

REVIEW_CATEGORIES = {"triggers", "needs_review", "rejected_transcript"}
MAX_AUDIO_BYTES = 4 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A Collector token must never follow an HTTP redirect to another host.
        return None


def _get(url: str, token: str):
    request = urllib.request.Request(url, headers={"X-Wakeword-Token": token})
    return urllib.request.build_opener(_NoRedirect()).open(request, timeout=60)


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
            body = response.read(8 * 1024 * 1024 + 1)
            if len(body) > 8 * 1024 * 1024:
                raise ProjectError("Collector-Prüfliste ist zu groß")
            listing = json.loads(body)
    except urllib.error.HTTPError as err:
        if err.code == 404:
            raise ProjectError("Collector-Prüfschnittstelle fehlt; Wake Word Collector in HACS aktualisieren") from err
        raise ProjectError(f"Collector-Prüfabruf fehlgeschlagen (HTTP {err.code})") from err
    if not isinstance(listing, dict) or listing.get("labels_confirmed") is not False:
        raise ProjectError("Collector antwortet nicht mit einem unbestätigten Prüfbestand")
    items = listing.get("clips", [])
    if not isinstance(items, list) or len(items) > limit:
        raise ProjectError("Ungültiger Prüfbestand vom Collector")
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
) -> dict:
    from .audit import run as audit

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
        )
        by_hash = {item["sha256"]: item for item in snapshot["clips"]}
        evidence = []
        for line in (output / "audit" / "clips.jsonl").read_text().splitlines():
            item = json.loads(line)
            origin = by_hash[item["sha256"]]
            evidence.append(
                {key: origin.get(key) for key in ("category", "device", "filename", "sha256", "created_at")}
                | {key: item.get(key) for key in ("transcript", "findings", "scores", "teachers", "error")}
            )
        result = {
            **summary,
            "clips": evidence,
            "source": "collector.review",
            "truncated": snapshot["truncated"],
            "labels_changed": False,
            "output": str(output),
        }
    result.update(created_at=datetime.now(UTC).isoformat(), run_id=output.name)
    target = project.path("review.json")
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    temporary.chmod(0o600)
    temporary.replace(target)
    return result
