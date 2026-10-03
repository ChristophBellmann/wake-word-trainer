"""Synthetic speech with Piper voices (package piper-sample-generator).

Synthetic samples add many voices and speaking speeds that your own
recordings cannot cover. They are only used for training, never for the
evaluation: a model that is good on synthetic speech is not necessarily good
on yours.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from . import audio
from .project import Project, ProjectError


def voices(project: Project) -> list[Path]:
    paths = [project.resolve(voice) for voice in project.config["tts"]["voices"]]
    if not paths:
        raise ProjectError(
            "tts.voices is empty: add Piper voices, e.g. wake-word-trainer download --voice en_US-lessac-medium"
        )
    for path in paths:
        if not path.is_file():
            raise ProjectError(f"Voice missing: {path}")
        if path.suffix == ".onnx" and not Path(f"{path}.json").is_file():
            raise ProjectError(f"Voice config missing: {path}.json")
    return paths


def _per_phrase(total: int, phrases: list[str]) -> list[int]:
    base, extra = divmod(total, len(phrases))
    return [base + (1 if index < extra else 0) for index in range(len(phrases))]


def generate_set(
    project: Project, phrases: list[str], total: int, target: Path, runner=subprocess.run, force: bool = False
) -> int:
    """Generate `total` samples spread over the phrases into target/<n>/. Existing
    sets are kept unless force; returns the number of WAVs."""
    if not phrases or total <= 0:
        return 0
    if force and target.exists():
        shutil.rmtree(target)
    model_args: list[str] = []
    for voice in voices(project):
        model_args += ["--model", str(voice)]
    scales = [str(scale) for scale in project.config["tts"]["length_scales"]]
    for index, (phrase, count) in enumerate(zip(phrases, _per_phrase(total, phrases), strict=True)):
        folder = target / f"{index:02d}"
        if count == 0 or len(audio.wavs(folder)) >= count:
            continue
        command = [
            sys.executable,
            "-m",
            "piper_sample_generator",
            phrase,
            *model_args,
            "--max-samples",
            str(count),
            "--output-dir",
            str(folder),
        ]
        if scales:
            command += ["--length-scales", *scales]
        result = runner(command, check=False)
        if result.returncode != 0:
            raise ProjectError(f"Piper failed for {phrase!r} (exit {result.returncode}); is the tts extra installed?")
    return len(audio.wavs(target))


def generate(project: Project, force: bool = False, runner=subprocess.run) -> tuple[int, int]:
    settings = project.config["tts"]
    positives = generate_set(
        project, list(settings["phrases"]), int(settings["samples"]), project.path("tts", "positive"), runner, force
    )
    negatives = generate_set(
        project,
        list(settings["negative_phrases"]),
        int(settings["negative_samples"]),
        project.path("tts", "negative"),
        runner,
        force,
    )
    return positives, negatives
