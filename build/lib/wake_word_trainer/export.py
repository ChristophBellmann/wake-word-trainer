"""The model for ESPHome: <slug>.tflite plus the manifest micro_wake_word reads."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .evaluate import Report
from .features import STEP_MS
from .project import Project, ProjectError


def manifest(project: Project, report: Report) -> dict:
    if report.probability_cutoff is None:
        raise ProjectError("The evaluation found no usable cutoff; nothing to export")
    settings = project.config["export"]
    return {
        "type": "micro",
        "wake_word": project.config["wake_word"],
        "author": settings["author"] or "wake-word-trainer",
        "website": settings["website"] or "https://github.com/ChristophBellmann/wake-word-trainer",
        "model": f"{project.slug}.tflite",
        "trained_languages": [project.config["language"]],
        "version": 2,
        "micro": {
            "probability_cutoff": report.probability_cutoff,
            "sliding_window_size": report.sliding_window_size,
            "feature_step_size": STEP_MS,
            "tensor_arena_size": int(settings["tensor_arena_size"]),
            "minimum_esphome_version": settings["minimum_esphome_version"],
        },
    }


def export(project: Project, model_path: Path, report: Report) -> Path:
    data = manifest(project, report)
    project.export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, project.export_dir / data["model"])
    target = project.export_dir / f"{project.slug}.json"
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return target


def esphome_snippet(manifest_path: Path) -> str:
    return (
        "micro_wake_word:\n"
        "  models:\n"
        f"    - model: {manifest_path}   # or an http(s) URL to the .json; the .tflite must be beside it\n"
    )
