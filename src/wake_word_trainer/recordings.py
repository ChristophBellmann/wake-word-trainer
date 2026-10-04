"""Your own recordings of the wake word: the most valuable training data.

Sources:
* the export of the Home Assistant integration Wake Word Collector (only
  clips accepted there; rejected clips disappear here on the next fetch),
* local folders of WAV files.

Each clip is assigned to `train` or `eval` by a hash of its content, so a clip
keeps its side on every fetch and evaluation never sees training material.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import audio
from .project import Project, ProjectError

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass
class FetchResult:
    added: int = 0
    removed: int = 0
    kept: int = 0
    train: int = 0
    eval: int = 0
    hard: int = 0
    negative: int = 0

    def __str__(self) -> str:
        text = (
            f"{self.added} new, {self.removed} removed, {self.kept} unchanged; "
            f"{self.train} for training, {self.eval} held out for evaluation"
        )
        if self.hard:
            text += f"; {self.hard} difficult"
        if self.negative:
            text += f"; {self.negative} without the wake word"
        return text


def split_for(digest: str, eval_share: float) -> str:
    """Stable assignment: the same content always lands on the same side."""
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < eval_share else "train"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get(url: str, token: str, timeout: float = 60):
    request = urllib.request.Request(url, headers={"X-Wakeword-Token": token})
    return urllib.request.urlopen(request, timeout=timeout)


def read_token(project: Project) -> str:
    token_file = project.config["collector"]["token_file"]
    if not token_file:
        raise ProjectError("collector.token_file is not set")
    path = project.resolve(token_file)
    if not path.is_file():
        raise ProjectError(f"Token file missing: {path}")
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ProjectError(f"Token file is empty: {path}")
    return token


def _listing(base: str, token: str, path: str, key: str, optional: bool = False) -> list[dict]:
    try:
        with _get(base + path, token) as response:
            return list(json.load(response).get(key, []))
    except urllib.error.HTTPError as err:
        if optional and err.code == 404:
            return []  # older collector without this export
        raise


def collector_clips(base_url: str, token: str, slug: str, kind: str = "positive") -> list[tuple[str, str, bytes]]:
    """(device, filename, wav bytes) from the collector export: the accepted clips
    (kind positive) or the clips marked as not the wake word (kind negative)."""
    base = base_url.rstrip("/")
    if urllib.parse.urlsplit(base).scheme not in {"http", "https"}:
        raise ProjectError("collector.url must start with http:// or https://")
    quoted = urllib.parse.quote(slug)
    if kind == "negative":
        items = _listing(base, token, f"/api/wake_word_collector/export/{quoted}/negatives", "negatives", True)
    else:
        items = _listing(base, token, f"/api/wake_word_collector/export/{quoted}", "candidates")
    clips = []
    for item in items:
        device, filename = str(item.get("device", "")), str(item.get("filename", ""))
        if not SAFE_NAME.match(device) or not SAFE_NAME.match(filename) or not filename.endswith(".wav"):
            continue
        with _get(base + str(item["audio_url"]), token) as response:
            clips.append((device, filename, response.read()))
    return clips


def local_clips(project: Project, setting: str = "folders") -> list[tuple[str, str, bytes]]:
    clips = []
    for folder in project.config["recordings"][setting]:
        root = project.resolve(folder)
        if not root.is_dir():
            raise ProjectError(f"Recording folder missing: {root}")
        for path in audio.wavs(root):
            data = path.read_bytes()
            clips.append(("local", f"local_{_sha256(data)[:12]}.wav", data))
    return clips


def fetch(project: Project) -> FetchResult:
    """Bring recordings/ in line with the sources; returns what changed.

    recordings/train|eval/<device>/           the wake word
    recordings/hard/<device>/                 difficult examples, training only
    recordings/negative/train|eval/<device>/  no wake word
    """
    collector = project.config["collector"]
    positives, negatives = [], []
    if collector["url"]:
        token = read_token(project)
        positives += collector_clips(collector["url"], token, project.slug)
        negatives += collector_clips(collector["url"], token, project.slug, "negative")
    positives += local_clips(project)
    hard = local_clips(project, "hard_folders")
    negatives += local_clips(project, "negative_folders")
    if not (positives or hard):
        raise ProjectError("No recordings: set collector.url/token_file or recordings.folders")

    share = float(project.config["recordings"]["eval_share"])
    root = project.recordings
    wanted: dict[Path, bytes] = {}
    held_out = set()
    for device, filename, data in positives:
        digest = _sha256(data)
        split = split_for(digest, share)
        wanted[root / split / device / filename] = data
        if split == "eval":
            held_out.add(digest)
    for device, filename, data in hard:
        # A difficult-example folder may repeat ordinary recordings. Evaluation
        # always wins: keep the clip held out, regardless of its source/name.
        if _sha256(data) in held_out:
            continue
        wanted[root / "hard" / device / filename] = data
    for device, filename, data in negatives:
        wanted[root / "negative" / split_for(_sha256(data), share) / device / filename] = data

    result = FetchResult()
    for path in audio.wavs(root):
        if path not in wanted:
            path.unlink()
            result.removed += 1
    for path, data in wanted.items():
        if path.is_file() and path.read_bytes() == data:
            result.kept += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        result.added += 1
    for folder in sorted(root.rglob("*"), reverse=True):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    result.train = len(audio.wavs(root / "train"))
    result.eval = len(audio.wavs(root / "eval"))
    result.hard = len(audio.wavs(root / "hard"))
    result.negative = len(audio.wavs(root / "negative"))
    return result
