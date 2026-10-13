"""wake-word-trainer: from your recordings to a wake word model on your ESPHome satellites."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .project import Project, ProjectError


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
    for preset in args.speech or []:
        clips = int(project.config["negatives"]["speech_clips"])
        path = downloads.speech(
            preset, target, clips, exclude_phrases=[project.config["wake_word"], *project.config["tts"]["phrases"]]
        )
        print(f"Speech {path}; add {preset!r} to negatives.speech_presets in wakeword.yaml")
    if (args.voice or args.speech) and not args.data:
        return
    sets = args.negative_set or project.config["negatives"]["feature_sets"]
    downloads.negative_features(target, sets=sets)
    if args.negative_set:
        project.config["negatives"]["feature_sets"] = sets
        project.path("wakeword.yaml").write_text(project.render_config(), encoding="utf-8")
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


def cmd_data(args: argparse.Namespace) -> None:
    from .datasets import import_sources, preflight

    project = Project.load(args.project)
    result = (
        import_sources(project, Path(args.manifest).expanduser().resolve(), args.dry_run)
        if args.data_command == "import"
        else preflight(project, rehash=args.rehash)
    )
    print(json.dumps(result, ensure_ascii=False))


def cmd_experiment(args: argparse.Namespace) -> None:
    from .experiments import prepare, run

    project = Project.load(args.project)
    output = project.resolve(args.output)
    if args.experiment_command == "prepare":
        projects = prepare(project, output, args.seed, args.model_profile)
        print(f"{len(projects)} unabhängige Läufe vorbereitet: {output}")
    else:
        print(f"{len(run(output, _downloads(project, args)))} Läufe bewertet: {output / 'results.json'}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from .benchmark import run
    from .state import State, now

    project = Project.load(args.project)
    output = project.resolve(args.output)
    state = State(project)
    state.update(state="running", phase="benchmark", started_at=now(), message="GPU-Durchsatz wird gemessen",
                 best_model_available=False)
    try:
        report = run(
            project,
            _downloads(project, args),
            output,
            engines=args.engine or ["standard", "optimized"],
            pipelines=args.pipeline or ["python", "prefetch", "tfrecord"],
            batch_sizes=args.batch_size or [128, 256],
            steps=args.steps,
            warmup=args.warmup,
            repeats=args.repeats,
            device=args.device,
            model_profile=args.model_profile,
            input_threads=args.input_threads,
            validation_cache_mb=args.validation_cache_mb,
            validation_batch_size=args.validation_batch_size,
        )
        project.path("benchmark.json").write_text(
            json.dumps({"report": str(output / "summary.json"), **report}, indent=2)
        )
        state.update(
            state="completed", phase="done", ended_at=now(), exit_code=0, message=f"GPU-Benchmark fertig: {output}",
            best_model_available=False,
        )
    except BaseException as err:
        state.update(state="failed", ended_at=now(), last_error=str(err))
        raise
    print(f"Benchmark: {output / 'README.md'}")


def cmd_audit(args: argparse.Namespace) -> None:
    from .audit import run

    project = Project.load(args.project)
    extra = [
        (label, project.resolve(folder))
        for label in ("positive", "negative", "unlabelled")
        for folder in getattr(args, label + "_folder")
    ]
    result = run(
        project,
        extra,
        project.resolve(args.output),
        args.stt_model,
        [project.resolve(path) for path in args.model],
        args.phrase,
        args.threads,
        [project.resolve(path) for path in args.teacher],
    )
    print(f"{result['unique_clips']} Aufnahmen geprüft; Labels unverändert. Bericht: {project.resolve(args.output)}")
    print(json.dumps(result["findings"], ensure_ascii=False))
    if result["findings"].get("processing_error"):
        raise ProjectError("Einzelne Aufnahmen konnten nicht geprüft werden; Details stehen im privaten Bericht")


def cmd_teacher(args: argparse.Namespace) -> None:
    from .teacher import Teacher, import_checkpoint, train

    project = Project.load(args.project)
    if args.teacher_command == "import":
        target = project.resolve(args.output)
        import_checkpoint(project.resolve(args.checkpoint), project.resolve(args.config), target)
        print(f"Portabler Teacher: {target}")
    elif args.teacher_command == "train":
        checkpoint = train(project, args.epochs, args.batch_size, args.seed, device=args.device,
                           include_corpus=args.include_corpus, negative_limit=args.negative_limit,
                           initial_model=project.resolve(args.initial_model) if args.initial_model else None,
                           balanced_sources=args.balanced_sources, architecture=args.architecture)
        print(f"Teacher gespeichert: {checkpoint}")
    else:
        from . import audio
        from .audit import inventory

        scorer = Teacher(project.resolve(args.model), args.threads, device=args.device)
        rows = []
        for item in inventory(project, [("unlabelled", project.resolve(p)) for p in args.folder]):
            rows.append({**item, "teacher": scorer(audio.read(item["sources"][0]["path"]))})
            if len(rows) % 25 == 0:
                print(f"Teacher: {len(rows)} Aufnahmen bewertet", flush=True)
        target = project.resolve(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"model": scorer.metadata, "clips": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        target.chmod(0o600)
        print(f"{len(rows)} Teacher-Bewertungen: {target}")


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
    from .pipeline import latest_model

    return latest_model(project)


def _print_report(report) -> None:
    print(report.summary())
    if report.missed:
        print(
            f"Not recognized ({len(report.missed)}): listen to them; mislabelled clips belong rejected in the collector"
        )
        for path in report.missed[:20]:
            print(f"  {path}")
    if report.triggered_by:
        print(f"Own recordings that wrongly trigger ({len(report.triggered_by)}):")
        for path in report.triggered_by[:20]:
            print(f"  {path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .evaluate import evaluate

    project = Project.load(args.project)
    _print_report(evaluate(project, _model(project), _downloads(project, args)))


def cmd_compare(args: argparse.Namespace) -> None:
    from .evaluate import compare

    project = Project.load(args.project)
    models = [Path(model).expanduser().resolve() for model in args.model]
    if args.current:
        models.insert(0, project.export_dir / f"{project.slug}.tflite")
    for model in models:
        if not model.is_file():
            raise ProjectError(f"Model missing: {model}")
    rows = compare(project, models, _downloads(project, args))
    for row in rows:
        print(f"\n{row['model']}")
        budget = row["budget"]
        print(f"  within the budget:  {_line(budget)}")
        if "own_cutoff" in row:
            print(f"  at its own cutoff:  {_line(row['own_cutoff'])}")
    print(f"\nDetails: {project.path('compare.json')}")


def _line(report: dict) -> str:
    if report["probability_cutoff"] is None:
        return f"no cutoff within {report['max_false_accepts_per_hour']} false activations per hour"
    text = (
        f"cutoff {report['probability_cutoff']:.3f}, recall {report['recall']:.1%} of {report['positives']}, "
        f"{report['false_accepts_per_hour']:.2f}/h"
    )
    if report.get("own_negatives"):
        text += f", {report['own_negatives_triggered']}/{report['own_negatives']} own negatives trigger"
    return text


def cmd_mine(args: argparse.Namespace) -> None:
    from . import mining

    project = Project.load(args.project)
    rounds = sorted(project.path("rounds").glob("round*.tflite"))
    number = len(rounds) + 1
    mining.mine(project, _model(project), _downloads(project, args), number)
    print("Train again (train) to use the mined examples.")


def cmd_status(args: argparse.Namespace) -> None:
    from .state import read

    project = Project.load(args.project)
    print(json.dumps(read(project), indent=2))


def cmd_export(args: argparse.Namespace) -> None:
    from .evaluate import Report
    from .export import esphome_snippet, export

    project = Project.load(args.project)
    report_path = project.path("report.json")
    if not report_path.is_file():
        raise ProjectError("No evaluation: run evaluate first")
    data = json.loads(report_path.read_text(encoding="utf-8"))
    data.pop("parity", None)
    report = Report(**data)
    manifest = export(project, _model(project), report)
    print(f"Exported {manifest} and {manifest.with_suffix('.tflite')}\n")
    print(esphome_snippet(manifest))


def cmd_run(args: argparse.Namespace) -> None:
    from .export import esphome_snippet
    from .pipeline import run

    project = Project.load(args.project)
    report = run(
        project,
        _downloads(project, args),
        resume=args.resume,
        force_tts=args.force,
        rounds=args.rounds,
        steps=args.steps,
        profile=args.profile or "",
        prepare_only=args.prepare_only,
    )
    if report is None:
        return
    print()
    print(esphome_snippet(project.export_dir / f"{project.slug}.json"))


def cmd_deploy(args: argparse.Namespace) -> None:
    from .deployment import assess, deploy

    project = Project.load(args.project)
    if args.compare:
        parity = assess(project, _downloads(project, args))
        print(parity["reason"])
    if not deploy(project):
        raise ProjectError("Deployment blocked or failed; see deployment.json and deployment.log")
    print("Firmware rollout completed")


def cmd_serve(args: argparse.Namespace) -> None:
    from .service import serve

    serve(Path(args.config).expanduser())


COMMANDS = {
    "init": cmd_init,
    "download": cmd_download,
    "fetch": cmd_fetch,
    "data": cmd_data,
    "experiment": cmd_experiment,
    "benchmark": cmd_benchmark,
    "audit": cmd_audit,
    "teacher": cmd_teacher,
    "generate": cmd_generate,
    "features": cmd_features,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "export": cmd_export,
    "run": cmd_run,
    "compare": cmd_compare,
    "mine": cmd_mine,
    "status": cmd_status,
    "serve": cmd_serve,
    "deploy": cmd_deploy,
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
    download.add_argument(
        "--speech", action="append", help="more negative speech, e.g. mls_de (German audio books; repeatable)"
    )
    download.add_argument("--data", action="store_true", help="with --voice/--speech: also the shared data")
    download.add_argument("--negative-set", action="append", help="MMAP-Negativquelle, auch *_background; wiederholbar")

    sub.add_parser("fetch", help="own recordings from the collector and recordings.folders")
    data = sub.add_parser("data", help="Datenquellen mit Rollen übernehmen und vor dem Training prüfen")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_import = data_sub.add_parser("import")
    data_import.add_argument("manifest", help="YAML mit sources: [{folder, role}]")
    data_import.add_argument("--dry-run", action="store_true")
    data_check = data_sub.add_parser("check")
    data_check.add_argument("--rehash", action="store_true", help="Dateien auch bei unveränderten Metadaten neu hashen")
    experiment = sub.add_parser("experiment", help="getrennte Seed-Vergleiche mit festen Daten, ohne Rollout")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    experiment_prepare = experiment_sub.add_parser("prepare")
    experiment_prepare.add_argument("--seed", action="append", type=int, required=True)
    experiment_prepare.add_argument(
        "--model-profile", choices=["esphome_v2_like", "current", "current_wider"], default="current_wider"
    )
    experiment_prepare.add_argument("--output", default="experiments")
    experiment_run = experiment_sub.add_parser("run")
    experiment_run.add_argument("--output", default="experiments")
    benchmark = sub.add_parser(
        "benchmark", help="GPU-Durchsatz auf vorhandenen Features messen; ohne Export oder Rollout"
    )
    benchmark.add_argument("--output", default="benchmarks/latest")
    benchmark.add_argument("--device", choices=["gpu", "cpu", "auto"], default="gpu")
    benchmark.add_argument("--engine", choices=["standard", "optimized"], action="append")
    benchmark.add_argument("--pipeline", choices=["python", "prefetch", "tfrecord"], action="append")
    benchmark.add_argument("--batch-size", type=int, action="append")
    benchmark.add_argument("--warmup", type=int, default=25)
    benchmark.add_argument("--steps", type=int, default=100)
    benchmark.add_argument("--repeats", type=int, default=2)
    benchmark.add_argument("--input-threads", type=int)
    benchmark.add_argument("--validation-cache-mb", type=int)
    benchmark.add_argument("--validation-batch-size", type=int)
    benchmark.add_argument("--model-profile", choices=["esphome_v2_like", "current", "current_wider"])
    audit = sub.add_parser("audit", help="Positive, Negative und Auslösungen unabhängig mit STT prüfen")
    audit.add_argument("--output", default="audit", help="privater Berichtordner, relativ zum Projekt")
    audit.add_argument("--stt-model", default="small", help="lokales faster-whisper-Modell oder Modellname")
    audit.add_argument("--threads", type=int, default=4)
    audit.add_argument("--model", action="append", default=[], help="TFLite samt Manifest bewerten (wiederholbar)")
    audit.add_argument("--teacher", action="append", default=[], help="portablen Log-Mel-Teacher zusätzlich bewerten")
    audit.add_argument("--phrase", action="append", default=[], help="zusätzliche erlaubte STT-Schreibweise")
    for label in ("positive", "negative", "unlabelled"):
        audit.add_argument("--" + label + "-folder", action="append", default=[], help="zusätzliche WAV-Quelle")
    teacher = sub.add_parser("teacher", help="optionalen Log-Mel-Teacher importieren, prüfen oder trainieren")
    teacher_sub = teacher.add_subparsers(dest="teacher_command", required=True)
    teacher_import = teacher_sub.add_parser("import")
    teacher_import.add_argument("checkpoint")
    teacher_import.add_argument("--config", required=True, help="ursprüngliche Teacher-YAML mit Feature-Parametern")
    teacher_import.add_argument("--output", default="teacher/model.pt")
    teacher_train = teacher_sub.add_parser("train")
    teacher_train.add_argument("--epochs", type=int, default=30)
    teacher_train.add_argument("--batch-size", type=int, default=128)
    teacher_train.add_argument("--seed", type=int, default=1337)
    teacher_train.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    teacher_train.add_argument(
        "--include-corpus", action="store_true", help="TTS und getrennte WAV-Korpora einbeziehen"
    )
    teacher_train.add_argument("--negative-limit", type=int, default=6000, help="Maximalzahl je externem Negativkorpus")
    teacher_train.add_argument("--initial-model", help="Kompatiblen portablen Log-Mel-Teacher weitertrainieren")
    teacher_train.add_argument("--balanced-sources", action="store_true", help="Feste Quellanteile wie beim Student")
    teacher_train.add_argument("--architecture", choices=["legacy_v2", "temporal_v1"], default="legacy_v2")
    teacher_score = teacher_sub.add_parser("score")
    teacher_score.add_argument("--model", default="teacher/model.pt")
    teacher_score.add_argument("--folder", action="append", default=[])
    teacher_score.add_argument("--output", default="teacher/scores.json")
    teacher_score.add_argument("--threads", type=int, default=2)
    teacher_score.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    generate = sub.add_parser("generate", help="synthetic speech with Piper")
    generate.add_argument("--force", action="store_true", help="regenerate existing samples")
    sub.add_parser("features", help="augmented spectrograms")
    train = sub.add_parser("train", help="train the model")
    train.add_argument("--restart", action="store_true", help="do not resume from the last checkpoint")
    sub.add_parser("evaluate", help="recall on own recordings, false activations, cutoff")
    sub.add_parser("export", help="model and manifest for ESPHome")
    run = sub.add_parser("run", help="fetch, generate, features, training rounds, evaluate, export the best")
    run.add_argument("--force", action="store_true", help="regenerate synthetic speech")
    run.add_argument("--resume", action="store_true", help="continue from the last checkpoint instead of from scratch")
    run.add_argument("--steps", type=int, help="training steps per round (default: training.steps)")
    run.add_argument("--prepare-only", action="store_true", help="only fetch, generate and build features")
    run.add_argument("--profile", help=argparse.SUPPRESS)
    run.add_argument(
        "--rounds", type=int, help="mining rounds after the first training (default: training.mining_rounds)"
    )
    compare = sub.add_parser("compare", help="several models on the same evaluation data")
    compare.add_argument("--model", action="append", default=[], help=".tflite (manifest .json beside it is used)")
    compare.add_argument("--current", action="store_true", help="include this project's exported model")
    sub.add_parser("mine", help="collect what the trained model wrongly reacts to (then train again)")
    sub.add_parser("status", help="progress of the current or last run")
    deploy = sub.add_parser("deploy", help="retry the configured firmware rollout after a passed comparison")
    deploy.add_argument(
        "--compare", action="store_true", help="compare the current export against the deployed reference"
    )
    serve = sub.add_parser("serve", help="HTTP service for Home Assistant (see README)")
    serve.add_argument("--config", required=True, help="service configuration (YAML)")
    return main


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if (
        argv is None
        and args.command in {"run", "train", "teacher", "benchmark", "experiment", "serve", "audit"}
        and not os.environ.get("WAKE_WORD_TRAINER_RUNTIME_READY")
    ):
        from .gpu import environment

        prepared = environment()
        if any(prepared.get(key) != os.environ.get(key) for key in ("LD_LIBRARY_PATH", "LD_PRELOAD")):
            prepared["WAKE_WORD_TRAINER_RUNTIME_READY"] = "1"
            os.execve(sys.executable, [sys.executable, "-m", "wake_word_trainer", *sys.argv[1:]], prepared)
    try:
        COMMANDS[args.command](args)
    except ProjectError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
