"""The whole run: fetch, generate, features, training rounds with mining, evaluation, export.

Round 1 trains from scratch. Each further round first mines what the model
wrongly reacts to, then continues training with it. Every round is evaluated
on the same held-out data; the best round (highest recall within the false
activation budget, then fewest false activations) is exported.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from . import mining
from .evaluate import Report, evaluate
from .export import export
from .features import build
from .gpu import lease
from .project import Project, ProjectError
from .recordings import fetch
from .state import State, now
from .training import TFLITE, train
from .tts import generate

Log = Callable[[str], None]

CHECKPOINT_FILES = ("restore", "best_weights.weights.h5", "last_weights.weights.h5", "training_config.yaml")


def snapshot_checkpoint(source: Path, target: Path) -> None:
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINT_FILES:
        path = source / name
        if path.is_dir():
            shutil.copytree(path, target / name)
        elif path.is_file():
            shutil.copy2(path, target / name)


def restore_checkpoint(source: Path, target: Path) -> None:
    for name in CHECKPOINT_FILES:
        path = source / name
        destination = target / name
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        if not path.exists():
            continue
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)


def better(new: Report, old: Report | None) -> bool:
    if old is None or old.recall is None:
        return new.recall is not None
    if new.recall is None:
        return False
    if new.recall != old.recall:
        return new.recall > old.recall
    return (new.false_accepts_per_hour or 0) < (old.false_accepts_per_hour or 0)


def run(
    project: Project,
    downloads: Path,
    **kwargs,
) -> Report | None:
    hidden = any(
        os.environ.get(key) == "-1" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    )
    guard = (
        lease("training-pipeline")
        if not kwargs.get("prepare_only") and project.config["training"]["device"] != "cpu" and not hidden
        else contextlib.nullcontext()
    )
    with guard as handle:
        return _run(project, downloads, gpu_handle=handle, **kwargs)


def _run(
    project: Project,
    downloads: Path,
    *,
    resume: bool = False,
    force_tts: bool = False,
    rounds: int | None = None,
    steps: int | None = None,
    profile: str = "",
    prepare_only: bool = False,
    log: Log = print,
    gpu_handle=None,
) -> Report | None:
    settings = project.config["training"]
    if steps is not None:
        settings["steps"] = steps
    total = 1 + int(settings["mining_rounds"] if rounds is None else rounds)
    state = State(project)
    state.update(
        state="running",
        profile=profile,
        message="",
        last_error="",
        exit_code=None,
        started_at=now(),
        ended_at=None,
        round_current=1,
        rounds_total=total,
        step_current=0,
        steps_total=int(settings["steps"]),
        accepted_rounds=0,
        rejected_rounds=0,
        best_model_available=False,
        best_recall=None,
        best_faph=None,
        best_threshold=None,
        phase="fetch",
    )
    if project.config["deployment"]["enabled"]:
        from .deployment import record

        record(project, "pending", reason="Waiting for training and comparison")
        project.path("parity.json").unlink(missing_ok=True)
    try:
        if project.config["collector"]["url"] or any(
            project.config["recordings"][key] for key in ("folders", "hard_folders", "negative_folders")
        ):
            log(f"Recordings: {fetch(project)}")
        if project.config["tts"]["voices"]:
            state.update(phase="generate")
            positives, negatives = generate(project, force=force_tts)
            log(f"Synthetic speech: {positives} wake word, {negatives} look-alikes")
        state.update(phase="features")
        counts = build(project, downloads, log)
        if prepare_only:
            message = "Data ready: " + ", ".join(f"{name} {count}" for name, count in counts.items())
            state.update(state="completed", phase="done", ended_at=now(), exit_code=0, message=message)
            log(message)
            return None

        rounds_dir = project.model_dir.parent / "rounds"
        accepted_checkpoint = project.path("checkpoints", "accepted")
        if not resume:
            mining.reset(project)
            shutil.rmtree(rounds_dir, ignore_errors=True)
            shutil.rmtree(accepted_checkpoint, ignore_errors=True)
        rounds_dir.mkdir(parents=True, exist_ok=True)
        best: Report | None = None
        best_round = 0
        for number in range(1, total + 1):
            state.update(round_current=number, step_current=0)
            if number > 1:
                state.update(phase="mine")
                mining.mine(project, rounds_dir / f"round{best_round:02d}.tflite", downloads, number, log)
            state.update(phase="train")
            model = train(
                project,
                downloads,
                resume=resume or number > 1,
                on_step=lambda step: state.update(step_current=step),
                gpu_handle=gpu_handle,
            )
            state.update(phase="evaluate", step_current=int(settings["steps"]))
            report = evaluate(project, model, downloads)
            log(f"Round {number}: {report.summary()}")
            shutil.copyfile(model, rounds_dir / f"round{number:02d}.tflite")
            (rounds_dir / f"round{number:02d}.json").write_text(json.dumps(asdict(report), indent=2))
            if best is None and report.probability_cutoff is None:
                raise ProjectError(report.summary())
            if better(report, best):
                best, best_round = report, number
                snapshot_checkpoint(project.model_dir, accepted_checkpoint)
                state.update(accepted_rounds=int(state.data["accepted_rounds"]) + 1)
            else:
                # Die nächste Runde beginnt beim angenommenen Modell, wie im Lab.
                restore_checkpoint(accepted_checkpoint, project.model_dir)
                if best_round:
                    shutil.copyfile(rounds_dir / f"round{best_round:02d}.tflite", project.model_dir / TFLITE)
                state.update(rejected_rounds=int(state.data["rejected_rounds"]) + 1)
            state.update(
                best_recall=best.recall if best else None,
                best_faph=best.false_accepts_per_hour if best else None,
                best_threshold=best.probability_cutoff if best else None,
            )

        assert best is not None
        state.update(phase="export")
        best_model = rounds_dir / f"round{best_round:02d}.tflite"
        project.path("report.json").write_text(json.dumps(asdict(best), indent=2), encoding="utf-8")
        if best.probability_cutoff is None:
            raise ProjectError(best.summary())
        if not best.recall:
            raise ProjectError(f"The model recognizes none of the held-out recordings; not exported. {best.summary()}")
        manifest = export(project, best_model, best)
        if project.config["deployment"]["enabled"]:
            from .deployment import assess, record

            state.update(phase="parity")
            try:
                parity = assess(project, downloads)
                record(project, "ready" if parity["passed"] else "blocked", reason=parity["reason"])
                log(f"Deployment parity: {parity['reason']}")
            except Exception as err:
                # A comparison failure must block flashing, while preserving the trained model.
                record(project, "blocked", reason=f"Comparison failed ({type(err).__name__})")
                project.path("parity.json").unlink(missing_ok=True)
                log(f"Deployment blocked: comparison failed ({type(err).__name__})")
        message = f"Round {best_round} of {total}: {best.summary()}"
        state.update(
            state="completed",
            phase="done",
            ended_at=now(),
            exit_code=0,
            message=message,
            best_model_available=True,
            manifest=str(manifest),
        )
        log(message)
        return best
    except KeyboardInterrupt:
        state.update(state="stopped", ended_at=now(), exit_code=130, message="Stopped")
        raise
    except Exception as err:
        state.update(state="failed", ended_at=now(), exit_code=1, last_error=str(err))
        raise


def latest_model(project: Project) -> Path:
    model = project.model_dir / TFLITE
    if not model.is_file():
        raise ProjectError("No trained model: run train first")
    return model
