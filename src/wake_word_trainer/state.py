"""Progress of a run in <project>/state.json, read by `status` and the service."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from .project import Project

# Share of the whole run before training starts and after it ends.
PREPARE_SHARE = 10.0
FINISH_SHARE = 5.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class State:
    def __init__(self, project: Project):
        self.path = project.path("state.json")
        self.data: dict[str, Any] = read(project)

    def update(self, **fields: Any) -> None:
        self.data.update(fields)
        self.data["updated_at"] = now()
        rounds = max(1, int(self.data.get("rounds_total") or 1))
        done = int(self.data.get("round_current") or 1) - 1
        steps = int(self.data.get("steps_total") or 0)
        step = int(self.data.get("step_current") or 0)
        phase = self.data.get("phase")
        if phase in ("fetch", "generate", "features"):
            percent = PREPARE_SHARE * {"fetch": 0.1, "generate": 0.3, "features": 0.6}[phase]
        elif phase in ("train", "mine", "evaluate"):
            fraction = (done + (step / steps if steps else 0.0)) / rounds
            percent = PREPARE_SHARE + (100 - PREPARE_SHARE - FINISH_SHARE) * min(1.0, fraction)
        elif phase == "export":
            percent = 100 - FINISH_SHARE
        else:
            percent = self.data.get("progress_percent", 0)
        if self.data.get("state") == "completed":
            percent = 100
        self.data["progress_percent"] = round(percent, 1)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)


def read(project: Project) -> dict[str, Any]:
    path = project.path("state.json")
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"state": "idle"}
