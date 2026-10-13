"""Getrennte Seed-Läufe auf einem eingefrorenen Datenbestand, ohne Rollout."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from .datasets import preflight
from .project import Project, ProjectError


def prepare(project: Project, output: Path, seeds: list[int], profile: str) -> list[Project]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ProjectError("Mindestens ein Seed, ohne Wiederholungen, erforderlich")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ProjectError("Seeds müssen ganze Zahlen in 0..4294967295 sein")
    preflight(project)
    output.mkdir(parents=True, exist_ok=True)
    projects = []
    for seed in seeds:
        root = output / f"{profile}_seed{seed}"
        if root.exists():
            raise ProjectError(f"Experiment existiert bereits: {root}")
    for seed in seeds:
        settings = {key: value for key, value in project.config.items() if key not in {"wake_word", "slug", "language"}}
        child = Project.create(
            output / f"{profile}_seed{seed}", project.config["wake_word"], project.config["language"], **settings
        )
        child.config["collector"] = {"url": "", "token_file": ""}
        for key in ("folders", "hard_folders", "negative_folders"):
            child.config["recordings"][key] = []
        child.config["tts"]["voices"] = []
        child.config["training"].update(seed=seed, model_profile=profile, mining_rounds=0)
        child.config["deployment"] = {"enabled": False, "reference_model": "", "command": []}
        for section, keys in (
            ("negatives", ("speech_folders", "background_folders")),
            ("augmentation", ("background_folders", "rir_folders")),
        ):
            for key in keys:
                child.config[section][key] = [str(project.resolve(p).resolve()) for p in project.config[section][key]]
        for folder in ("recordings", "tts"):
            source = project.path(folder)
            if source.is_dir():
                shutil.copytree(source, child.path(folder))
        child.validate()
        child.path("wakeword.yaml").write_text(child.render_config(), encoding="utf-8")
        child.path("dataset-origin.json").write_text(
            json.dumps(
                {
                    "project": str(project.root),
                    "seed": seed,
                    "profile": profile,
                    "dataset": json.loads(project.path("dataset-check.json").read_text()),
                },
                indent=2,
            )
        )
        projects.append(child)
    (output / "plan.json").write_text(json.dumps({"projects": [str(p.root) for p in projects]}, indent=2))
    return projects


def run(output: Path, downloads: Path, log=print) -> list[dict]:
    from .pipeline import run as train

    plan = json.loads((output / "plan.json").read_text())
    results = []
    for folder in plan["projects"]:
        project = Project.load(folder)
        if project.config["deployment"]["enabled"]:
            raise ProjectError("Experiment darf keinen automatischen Rollout aktivieren")
        log(f"Experiment: {project.root.name}")
        report = train(project, downloads, rounds=0, log=log)
        results.append({"project": folder, "seed": project.config["training"]["seed"], "report": asdict(report)})
        (output / "results.json").write_text(json.dumps(results, indent=2))
    return results
