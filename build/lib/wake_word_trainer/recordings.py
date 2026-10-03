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
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import audio
from .project import Project, ProjectError

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SPLITS = ("train", "eval")


@dataclass
class FetchResult:
    added: int = 0
    removed: int = 0
    kept: int = 0
    train: int = 0
    eval: int = 0

    def __str__(self) -> str:
        return (
            f"{self.added} new, {self.removed} removed, {self.kept} unchanged; "
            f"{self.train} for training, {self.eval} held out for evaluation"
        )


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


def collector_clips(base_url: str, token: str, slug: str) -> list[tuple[str, str, bytes]]:
    """(device, filename, wav bytes) of every usable clip in the collector export."""
    base = base_url.rstrip("/")
    if urllib.parse.urlsplit(base).scheme not in {"http", "https"}:
        raise ProjectError("collector.url must start with http:// or https://")
    with _get(f"{base}/api/wake_word_collector/export/{urllib.parse.quote(slug)}", token) as response:
        listing = json.load(response)
    clips = []
    for item in listing.get("candidates", []):
        device, filename = str(item.get("device", "")), str(item.get("filename", ""))
        if not SAFE_NAME.match(device) or not SAFE_NAME.match(filename) or not filename.endswith(".wav"):
            continue
        with _get(base + str(item["audio_url"]), token) as response:
            clips.append((device, filename, response.read()))
    return clips


def local_clips(project: Project) -> list[tuple[str, str, bytes]]:
    clips = []
    for folder in project.config["recordings"]["folders"]:
        root = project.resolve(folder)
        if not root.is_dir():
            raise ProjectError(f"Recording folder missing: {root}")
        for path in audio.wavs(root):
            data = path.read_bytes()
            clips.append(("local", f"local_{_sha256(data)[:12]}.wav", data))
    return clips


def fetch(project: Project) -> FetchResult:
    """Bring recordings/ in line with the sources; returns what changed."""
    clips: list[tuple[str, str, bytes]] = []
    collector = project.config["collector"]
    if collector["url"]:
        clips += collector_clips(collector["url"], read_token(project), project.slug)
    clips += local_clips(project)
    if not clips:
        raise ProjectError("No recordings: set collector.url/token_file or recordings.folders")

    share = float(project.config["recordings"]["eval_share"])
    wanted: dict[Path, bytes] = {}
    for device, filename, data in clips:
        wanted[project.recordings / split_for(_sha256(data), share) / device / filename] = data

    result = FetchResult()
    for path in audio.wavs(*(project.recordings / split for split in SPLITS)):
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
    for split in SPLITS:
        for folder in (project.recordings / split).glob("*"):
            if folder.is_dir() and not any(folder.iterdir()):
                shutil.rmtree(folder)
    result.train = len(audio.wavs(project.recordings / "train"))
    result.eval = len(audio.wavs(project.recordings / "eval"))
    return result
