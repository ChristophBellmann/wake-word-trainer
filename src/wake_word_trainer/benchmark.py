"""Isolierte, wiederholte Durchsatzvergleiche auf denselben echten Features."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .gpu import environment, lease
from .project import Project, ProjectError
from .training import config as training_config
from .training import model_args


def _run(
    project: Project,
    downloads: Path,
    output: Path,
    *,
    engines=("standard", "optimized"),
    pipelines=("python", "prefetch", "tfrecord"),
    batch_sizes=(128, 256),
    warmup=25,
    steps=100,
    repeats=2,
    device="gpu",
    model_profile=None,
    input_threads=None,
    validation_cache_mb=None,
    validation_batch_size=None,
    log=print,
    gpu_handle=None,
) -> dict:
    if min(warmup, steps, repeats, *batch_sizes) < 1 or device not in {"gpu", "cpu", "auto"}:
        raise ProjectError(
            "Benchmark benötigt positive Schritte, Wiederholungen und Batchgrößen sowie ein gültiges Gerät"
        )
    if not set(engines) <= {"standard", "optimized"} or not set(pipelines) <= {"python", "prefetch", "tfrecord"}:
        raise ProjectError("Unbekannte Benchmark-Engine oder Eingabezuführung")
    if not engines or not pipelines:
        raise ProjectError("Benchmark benötigt mindestens eine Engine und Eingabezuführung")
    if output.exists():
        raise ProjectError(f"Benchmark-Ausgabe existiert bereits; neuen Ordner wählen: {output}")
    base = training_config(project, downloads)
    base.update(seed=42, device=device, input_cache_dir=str(output / "input_cache"))
    if input_threads is not None:
        base["input_threads"] = input_threads
    if validation_cache_mb is not None:
        base["validation_cache_mb"] = validation_cache_mb
    if validation_batch_size is not None:
        base["validation_batch_size"] = validation_batch_size
    # Gleiche Masken-Einstellung in allen Varianten. Hier reiner Durchsatzvergleich.
    for name in ("time_mask_count", "freq_mask_count"):
        base[name] = [0]
    profile = model_profile or project.config["training"]["model_profile"]
    if profile not in {"esphome_v2_like", "current", "current_wider"}:
        raise ProjectError("Unbekanntes Benchmark-Modellprofil")
    output.mkdir(parents=True)
    rows = []
    started = time.monotonic()
    for repeat in range(repeats):
        variants = [(engine, pipeline, batch) for engine in engines for pipeline in pipelines for batch in batch_sizes]
        if repeat % 2:
            variants.reverse()  # Reihenfolgeeffekte wie thermische Erwärmung sichtbar machen.
        for engine, pipeline, batch in variants:
            folder = output / f"{engine}_{pipeline}_b{batch}_r{repeat + 1}"
            folder.mkdir()
            settings = {
                **base,
                "batch_size": batch,
                "engine": engine,
                "input_pipeline": pipeline,
                "train_dir": str(folder / "model"),
            }
            path = folder / "config.yaml"
            path.write_text(yaml.safe_dump(settings, sort_keys=False))
            command = [
                sys.executable,
                "-X",
                "faulthandler",
                "-m",
                "wake_word_trainer.benchmark_worker",
                "--config",
                str(path),
                "--output",
                str(folder / "result.json"),
                "--warmup",
                str(warmup),
                "--steps",
                str(steps),
                *model_args(profile),
            ]
            if gpu_handle is not None:
                command[5:5] = ["--gpu-lock-fd", str(gpu_handle.fileno())]
            log(f"GPU-Benchmark: {engine}, {pipeline}, Batch {batch}, Wiederholung {repeat + 1}/{repeats}")
            begin = time.monotonic()
            with (folder / "worker.log").open("w") as stream:
                process = subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=environment(),
                    pass_fds=(gpu_handle.fileno(),) if gpu_handle is not None else (),
                )
                try:
                    returncode = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait(timeout=30)
                    raise
            result = folder / "result.json"
            row = (
                json.loads(result.read_text())
                if result.is_file()
                else {"valid": False, "error": f"Worker beendet mit {returncode}; {folder / 'worker.log'}"}
            )
            row.update(
                engine=engine,
                pipeline=pipeline,
                batch_size=batch,
                repeat=repeat + 1,
                wall_seconds=time.monotonic() - begin,
            )
            if returncode:
                row["valid"] = False
            rows.append(row)
            (output / "results.json").write_text(json.dumps(rows, indent=2))
            if row.get("valid"):
                log(
                    f"  {row['steps_per_second']:.2f} Schritte/s, {row['examples_per_second']:.0f} Beispiele/s; "
                    f"GPU Ø {row['gpu'].get('mean_use_percent', 0):.1f}%"
                )
            else:
                log(f"  Fehlgeschlagen: {row.get('error', 'siehe worker.log')}")
    groups = {}
    for row in rows:
        if row.get("valid"):
            groups.setdefault((row["engine"], row["pipeline"], row["batch_size"]), []).append(row)
    summary = []
    for (engine, pipeline, batch), measurements in groups.items():
        summary.append(
            {
                "engine": engine,
                "pipeline": pipeline,
                "batch_size": batch,
                "repeats": len(measurements),
                "median_steps_per_second": statistics.median(r["steps_per_second"] for r in measurements),
                "median_examples_per_second": statistics.median(r["examples_per_second"] for r in measurements),
                "min_examples_per_second": min(r["examples_per_second"] for r in measurements),
                "max_examples_per_second": max(r["examples_per_second"] for r in measurements),
            }
        )
    summary.sort(key=lambda r: r["median_examples_per_second"], reverse=True)
    report = {
        "model_profile": profile,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "requested_repeats": repeats,
        "seconds": time.monotonic() - started,
        "variants": summary,
        "failed_runs": sum(not r.get("valid") for r in rows),
        "note": "Durchsatz ist kein Qualitätsnachweis; gleiche Prüfaufnahmen und Cutoffs verwenden.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# GPU-Durchsatzvergleich",
        "",
        f"Modellprofil: `{profile}`; {warmup} Aufwärmschritte, {steps} Messschritte, {repeats} Wiederholungen.",
        "",
        "| Engine | Zuführung | Batch | Schritte/s, Median | Beispiele/s, Median | Wiederholungen |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {r['engine']} | {r['pipeline']} | {r['batch_size']} | {r['median_steps_per_second']:.2f} | "
        f"{r['median_examples_per_second']:.0f} | {r['repeats']} |"
        for r in summary
    ]
    lines += [
        "",
        "Aufwärmen, Input-Vorbereitung, Training und kalte/warme Validierung stehen getrennt in den Einzelberichten. "
        "Keine Projektgewichte, Labels oder Exporte verändert.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    if not summary or report["failed_runs"]:
        raise ProjectError(f"{report['failed_runs']} Benchmark-Läufe fehlgeschlagen; {output / 'results.json'}")
    return report


def run(project: Project, downloads: Path, output: Path, **options) -> dict:
    with lease("gpu-benchmark-campaign") as handle:
        return _run(project, downloads, output, gpu_handle=handle, **options)
