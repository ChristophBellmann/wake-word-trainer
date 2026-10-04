"""Temporarily pause configured systemd user services to free training VRAM."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .project import ProjectError


class Resources:
    def __init__(self, project: Path, units: list[str]):
        if not isinstance(units, list) or any(
            not isinstance(unit, str) or not re.fullmatch(r"[\w.@:-]+\.service", unit) for unit in units
        ):
            raise ProjectError("pause_services must be a list of systemd user .service names")
        self.units = list(dict.fromkeys(units))
        self.path = project / "paused_services.json"
        self.paused = json.loads(self.path.read_text()) if self.path.exists() else []

    def _save(self) -> None:
        part = self.path.with_suffix(".tmp")
        part.write_text(json.dumps(self.paused))
        part.replace(self.path)

    def acquire(self) -> None:
        try:
            for unit in self.units:
                status = subprocess.run(
                    ["systemctl", "--user", "is-active", unit], capture_output=True, text=True, timeout=10
                )
                if status.stdout.strip() not in {"active", "inactive", "failed"}:
                    raise ProjectError(f"Cannot inspect training resource service: {unit}")
                if status.stdout.strip() != "active":
                    continue
                # Save before stopping, so a service restart can recover interrupted runs.
                self.paused.append(unit)
                self._save()
                subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True, timeout=30, check=True)
        except Exception:
            self.release()
            raise

    def release(self) -> None:
        for unit in reversed(self.paused.copy()):
            subprocess.run(["systemctl", "--user", "start", unit], capture_output=True, timeout=60, check=True)
            self.paused.remove(unit)
            self._save()
        self.path.unlink(missing_ok=True)
