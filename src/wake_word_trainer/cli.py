"""wake-word-trainer: from your recordings to a wake word model on your ESPHome satellites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .project import Project, ProjectError

STEPS = ("fetch", "generate", "features", "train", "evaluate", "export")


def _downloads(project: Project, args: argparse.Namespace) -> Path:
    return Path(args.downloads).expanduser().resolve() if args.downloads else project.downloads


def cmd_init(args: argparse.Namespace) -> None:
    settings: dict = {}
    if args.collector_url:
        settings["collector"] = {"url": args.collector_url, "token_file": args.token_file or ""}
    project = Project.create(args.project, args.wake_word, args.language, **settings)
    print(f"Created {project.root / 'wakeword.yaml'} for {project.config['wake_word']!r} ({project.slug}).")
    print("Next: add Piper voices (download --voice ...) and similar sounding phrases, then: run")


def cmd_download(args: argparse.Namespace) -> None:
    from . import downloads

    project = Project.load(args.project)
    target = _downloads(project, args)
    for name in args.voice or []:
        path = downloads.voice(name, project.path("voices"))
        print(f"Voice {path}; add it to tts.voices in wakeword.yaml")
    if args.voice and not args.data:
        return
    downloads.negative_features(target)
    downloads.room_impulses(target)
    downloads.background(target)
    print(f"Shared data ready in {target}")


def cmd_fetch(args: argparse.Namespace) -> None:
    from .recordings import fetch

    project = Project.load(args.project)
    result = fetch(project)
    print(f"Recordings: {result}")
    if result.eval < 5:
        print("Hint: fewer than 5 held-out recordings make the evaluation unreliable; record more.")


def cmd_generate(args: argparse.Namespace) -> None:
    from .tts import generate

    project = Project.load(args.project)
    positives, negatives = generate(project, force=args.force)
    print(f"Synthetic speech: {positives} wake word samples, {negatives} look-alike samples")


def cmd_features(args: argparse.Namespace) -> None:
    from .features import build

    project = Project.load(args.project)
    counts = build(project, _downloads(project, args))
    print("Features: " + ", ".join(f"{name} {count}" for name, count in counts.items()))


def cmd_train(args: argparse.Namespace) -> None:
    from .training import train

    project = Project.load(args.project)
    model = train(project, _downloads(project, args), resume=not args.restart)
    print(f"Model: {model}")


def _model(project: Project) -> Path:
    from .training import TFLITE

    model = project.model_dir / TFLITE
    if not model.is_file():
        raise ProjectError("No trained model: run train first")
    return model


def cmd_evaluate(args: argparse.Namespace):
    from .evaluate import evaluate

    project = Project.load(args.project)
    report = evaluate(project, _model(project), _downloads(project, args))
    print(report.summary())
    if report.missed:
        print(
            f"Not recognized ({len(report.missed)}): listen to them; mislabelled clips belong rejected in the collector"
        )
        for path in report.missed[:20]:
            print(f"  {path}")
    return report


def cmd_export(args: argparse.Namespace) -> None:
    from .evaluate import Report
    from .export import esphome_snippet, export

    project = Project.load(args.project)
    report_path = project.path("report.json")
    if not report_path.is_file():
        raise ProjectError("No evaluation: run evaluate first")
    report = Report(**json.loads(report_path.read_text(encoding="utf-8")))
    manifest = export(project, _model(project), report)
    print(f"Exported {manifest} and {manifest.with_suffix('.tflite')}\n")
    print(esphome_snippet(manifest))


def cmd_run(args: argparse.Namespace) -> None:
    project = Project.load(args.project)
    has_sources = project.config["collector"]["url"] or project.config["recordings"]["folders"]
    steps = [step for step in STEPS if step != "fetch" or has_sources]
    if not project.config["tts"]["voices"]:
        steps.remove("generate")
    for step in steps:
        print(f"== {step}")
        COMMANDS[step](args)


COMMANDS = {
    "init": cmd_init,
    "download": cmd_download,
    "fetch": cmd_fetch,
    "generate": cmd_generate,
    "features": cmd_features,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "export": cmd_export,
    "run": cmd_run,
}


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(prog="wake-word-trainer", description=__doc__)
    main.add_argument("--version", action="version", version=__version__)
    main.add_argument("-p", "--project", default=".", help="project folder with wakeword.yaml (default: .)")
    main.add_argument("--downloads", help="shared download folder (default: <project>/downloads)")
    sub = main.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create wakeword.yaml")
    init.add_argument("wake_word", help='e.g. "Hey Jarvis"')
    init.add_argument("--language", default="en", help="language code, e.g. de")
    init.add_argument("--collector-url", help="Home Assistant URL of the Wake Word Collector")
    init.add_argument("--token-file", help="file with the collector token")

    download = sub.add_parser("download", help="background audio, impulse responses, negative features; Piper voices")
    download.add_argument("--voice", action="append", help="Piper voice, e.g. de_DE-thorsten-medium (repeatable)")
    download.add_argument("--data", action="store_true", help="with --voice: download the shared data as well")

    sub.add_parser("fetch", help="own recordings from the collector and recordings.folders")
    generate = sub.add_parser("generate", help="synthetic speech with Piper")
    generate.add_argument("--force", action="store_true", help="regenerate existing samples")
    sub.add_parser("features", help="augmented spectrograms")
    train = sub.add_parser("train", help="train the model")
    train.add_argument("--restart", action="store_true", help="do not resume from the last checkpoint")
    sub.add_parser("evaluate", help="recall on own recordings, false activations, cutoff")
    sub.add_parser("export", help="model and manifest for ESPHome")
    run = sub.add_parser("run", help="fetch, generate, features, train, evaluate, export")
    run.add_argument("--force", action="store_true", help="regenerate synthetic speech")
    run.add_argument("--restart", action="store_true", help="train from scratch")
    return main


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        COMMANDS[args.command](args)
    except ProjectError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
