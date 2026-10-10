"""Are memory, CPU and VRAM free enough to train? Linux /proc and sysfs, best effort.

Pausing known GPU services frees VRAM, but anything else on the computer can
still take the memory a run needs: a training that has to swap barely moves.
The service checks before a run starts and keeps watching memory while it runs.

Configuration (service.yaml, all optional):

    resource_check:
      enabled: true
      min_available_memory_gb: 6     # MemAvailable before and during a run
      max_load_per_cpu: 0.75          # 1-minute load average / CPUs, before a run
      min_free_vram_gb: null          # after pausing services; null = not checked
      wait_seconds: 10                # how long freed VRAM may take to show up
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .project import ProjectError

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "min_available_memory_gb": 6.0,
    "max_load_per_cpu": 0.75,
    "min_free_vram_gb": None,
    "wait_seconds": 10,
}
GIB = 1024**3


def memory(meminfo: Path = Path("/proc/meminfo")) -> dict[str, float]:
    values = {}
    try:
        for line in meminfo.read_text().splitlines():
            key, _, rest = line.partition(":")
            values[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return {}
    if "MemAvailable" not in values:
        return {}
    result = {"available_memory_gb": round(values["MemAvailable"] / GIB, 1)}
    if "SwapTotal" in values and "SwapFree" in values:
        result["swap_used_gb"] = round((values["SwapTotal"] - values["SwapFree"]) / GIB, 1)
    return result


def vram(drm: Path = Path("/sys/class/drm")) -> dict[str, float]:
    """Free VRAM of the largest GPU: NVIDIA via nvidia-smi, AMD via amdgpu sysfs."""
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        try:
            used, total = max(
                (tuple(float(v) for v in line.split(",")) for line in result.stdout.splitlines()),
                key=lambda pair: pair[1],
            )
            return {"free_vram_gb": round((total - used) / 1024, 1), "total_vram_gb": round(total / 1024, 1)}
        except ValueError:
            return {}
    cards = []
    for device in sorted(drm.glob("card*/device")):
        try:
            total = int((device / "mem_info_vram_total").read_text())
            used = int((device / "mem_info_vram_used").read_text())
        except (OSError, ValueError):
            continue
        cards.append((total, used))
    if not cards:
        return {}
    total, used = max(cards)
    return {"free_vram_gb": round((total - used) / GIB, 1), "total_vram_gb": round(total / GIB, 1)}


def load_per_cpu() -> float:
    return round(os.getloadavg()[0] / (os.cpu_count() or 1), 2)


class SystemCheck:
    def __init__(self, config: dict[str, Any] | None):
        unknown = set(config or {}) - set(DEFAULTS)
        if unknown:
            raise ProjectError(f"resource_check: unknown settings {sorted(unknown)}")
        self.config = {**DEFAULTS, **(config or {})}
        for key in ("min_available_memory_gb", "max_load_per_cpu", "min_free_vram_gb", "wait_seconds"):
            value = self.config[key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
                raise ProjectError(f"resource_check.{key} must be a non-negative number or null")

    def snapshot(self, with_vram: bool = True) -> dict[str, float]:
        result: dict[str, float] = {**memory(), "load_per_cpu": load_per_cpu()}
        if with_vram and self.config["min_free_vram_gb"] is not None:
            result.update(vram())
        return result

    def problems(self, snapshot: dict[str, float], phase: str) -> list[str]:
        """phase "start": before services are paused; "vram": after; "running": during a run."""
        if not self.config["enabled"]:
            return []
        found = []
        minimum = self.config["min_available_memory_gb"]
        if minimum is not None and phase in ("start", "running"):
            available = snapshot.get("available_memory_gb")
            if available is None:
                found.append("available memory unknown")
            elif available < minimum:
                found.append(f"only {available} GB memory available, need {minimum} GB")
        limit = self.config["max_load_per_cpu"]
        if limit is not None and phase == "start" and snapshot["load_per_cpu"] > limit:
            found.append(f"CPU load {snapshot['load_per_cpu']} per CPU, limit {limit}")
        needed = self.config["min_free_vram_gb"]
        if needed is not None and phase == "vram":
            free = snapshot.get("free_vram_gb")
            if free is None:
                found.append("free VRAM unknown")
            elif free < needed:
                found.append(f"only {free} GB VRAM free, need {needed} GB")
        return found

    def before_start(self) -> list[str]:
        return self.problems(self.snapshot(with_vram=False), "start")

    def after_pause(self, sleep: Callable[[float], None] = time.sleep) -> list[str]:
        """Freed VRAM can take a moment to show up after services stop."""
        if not self.config["enabled"] or self.config["min_free_vram_gb"] is None:
            return []
        deadline = time.monotonic() + float(self.config["wait_seconds"])
        while True:
            found = self.problems(self.snapshot(), "vram")
            if not found or time.monotonic() >= deadline:
                return found
            sleep(1)

    def while_running(self) -> list[str]:
        return self.problems(self.snapshot(with_vram=False), "running")
