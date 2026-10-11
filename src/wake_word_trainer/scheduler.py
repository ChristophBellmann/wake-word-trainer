"""Train changed collector data during a configured quiet night window.

Only runs started by this scheduler are interrupted when the user returns.
Unavailable activity or GPU probes fail closed. Commands and policy stay local.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import urllib.parse
from datetime import datetime, time, timedelta

from .idle import PREFIX
from .project import Project, ProjectError
from .recordings import _get, read_token
from .state import now, read

DEFAULTS = {
    "enabled": False,
    "start": "01:00",
    "end": "06:00",
    "idle_seconds": 1200,
    "min_new_samples": 5,
    "min_interval_hours": 20,
    "profile": "recommended",
    "check_seconds": 30,
    "idle_command": [],
    "max_gpu_use_percent": 5,
    "max_load_per_cpu": 0.5,
}


def in_window(current: time, start: str, end: str) -> bool:
    first, last = time.fromisoformat(start), time.fromisoformat(end)
    return first <= current < last if first < last else current >= first or current < last


class IdleProbeError(RuntimeError):
    """The idle command failed; its message is the probe's own reason line."""


class Scheduler:
    def __init__(self, service, config: dict):
        self.service = service
        self.config = {**DEFAULTS, **config}
        self.status = {"enabled": bool(self.config["enabled"]), "state": "disabled"}
        self.process = None
        self.pending: list[str] = []
        self.stop_event = threading.Event()
        self.thread = None
        self.path = service.project.path("automatic_training.json")
        self.saved = json.loads(self.path.read_text()) if self.path.is_file() else {}
        if self.config["enabled"]:
            if not self.config["idle_command"] or not isinstance(self.config["idle_command"], list):
                raise ProjectError("automatic_training.idle_command must print idle seconds")
            time.fromisoformat(self.config["start"])
            time.fromisoformat(self.config["end"])
            if self.config["start"] == self.config["end"]:
                raise ProjectError("automatic_training window must not cover the whole day")
            if self.config["profile"] not in service.profiles or service.profiles[self.config["profile"]].get(
                "prepare_only"
            ):
                raise ProjectError("automatic_training.profile must be a training profile")
            for key in ("idle_seconds", "min_new_samples", "min_interval_hours", "check_seconds"):
                if float(self.config[key]) <= 0:
                    raise ProjectError(f"automatic_training.{key} must be positive")
            self.status["state"] = "waiting"

    def _save(self):
        part = self.path.with_suffix(".tmp")
        part.write_text(json.dumps(self.saved, indent=2) + "\n")
        part.replace(self.path)

    def idle(self) -> float:
        result = subprocess.run(self.config["idle_command"], capture_output=True, text=True, timeout=10)
        if result.returncode:
            # Only the probe's own reason line is shown; other output stays private.
            reasons = [line for line in result.stderr.splitlines() if line.startswith(PREFIX)]
            raise IdleProbeError(reasons[-1] if reasons else f"idle command exited with {result.returncode}")
        value = float(result.stdout.strip())
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid idle seconds")
        return value

    def samples(self) -> set[str]:
        project = Project.load(self.service.project.root)
        base = project.config["collector"]["url"].rstrip("/")
        if not base or urllib.parse.urlsplit(base).scheme not in ("http", "https"):
            raise ProjectError("automatic_training requires a Collector URL")
        token, slug = read_token(project), urllib.parse.quote(project.slug)
        samples = set()
        for suffix, key in (("", "candidates"), ("/negatives", "negatives")):
            with _get(f"{base}/api/wake_word_collector/export/{slug}{suffix}", token, timeout=10) as response:
                data = json.load(response)
            for item in data.get(key, []):
                digest = item.get("sha256")
                if isinstance(digest, str) and len(digest) == 64:
                    samples.add(f"{key}:{digest}")
        return samples

    def tick(self, at: datetime | None = None):
        if not self.config["enabled"]:
            return
        at = at or datetime.now().astimezone()
        if "baseline" not in self.saved:
            self.saved["baseline"] = sorted(self.samples())
            self._save()
        night = in_window(at.time(), self.config["start"], self.config["end"])
        idle = self.idle()
        self.status.update(idle_seconds=round(idle), in_window=night)
        self.status.pop("problems", None)
        if self.process is not None:
            if self.process.poll() is not None:
                if read(self.service.project).get("state") == "completed" and self.process.returncode == 0:
                    self.saved["baseline"] = self.pending
                    self.saved["last_completed_at"] = now()
                    self._save()
                self.process = None
                self.pending = []
            elif not night or idle < float(self.config["idle_seconds"]):
                if self.service.process is self.process:
                    self.service.stop()
                self.process = None
                self.pending = []
                self.status["state"] = "interrupted_for_activity"
                return
            else:
                self.status["state"] = "training"
                return
        if not night or idle < float(self.config["idle_seconds"]):
            self.status["state"] = "outside_window" if not night else "user_active"
            return
        if self.service.running():
            self.status["state"] = "run_active"
            return
        deployment = self.service.project.path("deployment.json")
        if deployment.is_file() and json.loads(deployment.read_text()).get("state") == "running":
            self.status["state"] = "rollout_active"
            return
        last = self.saved.get("last_attempt_at")
        if last and at < datetime.fromisoformat(last) + timedelta(hours=float(self.config["min_interval_hours"])):
            self.status["state"] = "cooldown"
            return
        from .service import gpu

        metric = gpu()
        limit = self.config["max_gpu_use_percent"]
        if limit is not None and (metric.get("use_percent") is None or metric["use_percent"] > float(limit)):
            self.status["state"] = "gpu_busy_or_unknown"
            return
        if os.getloadavg()[0] / (os.cpu_count() or 1) > float(self.config["max_load_per_cpu"]):
            self.status["state"] = "cpu_busy"
            return
        samples = self.samples()
        if "baseline" not in self.saved:
            # Existing data has already been handled; enabling automation is not
            # permission to rerun every historical sample immediately.
            self.saved["baseline"] = sorted(samples)
            self._save()
        count = len(samples - set(self.saved["baseline"]))
        self.status["new_samples"] = count
        if count < int(self.config["min_new_samples"]):
            self.status["state"] = "waiting_for_samples"
            return
        if self.stop_event.is_set():
            return
        code, answer = self.service.start(self.config["profile"])
        if code == 202:
            self.process = self.service.process
            self.pending = sorted(samples)
            self.saved["last_attempt_at"] = at.isoformat()
            self._save()
            self.status["state"] = "training"
        elif code == 503 and answer.get("problems"):
            self.status["state"] = "resources_insufficient"
            self.status["problems"] = answer["problems"]
        else:
            self.status["state"] = "run_or_rollout_active"

    def _loop(self):
        while not self.stop_event.wait(float(self.config["check_seconds"])):
            try:
                self.tick()
                self.status.pop("error", None)
            except Exception as err:
                # Do not log responses, transcripts, credentials or probe output.
                error = str(err)[:200] if isinstance(err, IdleProbeError) else type(err).__name__
                self.status.update(state="probe_failed", error=error)
                if self.process is not None and self.process.poll() is None and self.service.process is self.process:
                    self.service.stop()
                    self.process = None

    def start(self):
        if self.config["enabled"]:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
