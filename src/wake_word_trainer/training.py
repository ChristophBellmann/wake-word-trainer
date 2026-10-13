"""Training with microWakeWord (MixedNet, streaming, int8 quantized)."""

from __future__ import annotations

import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import yaml

from .gpu import environment
from .project import Project, ProjectError

# MixedNet as used for the official ESPHome models.
MODEL_ARGS = [
    "mixednet",
    "--pointwise_filters",
    "64,64,64,64",
    "--repeat_in_block",
    "1,1,1,1",
    "--mixconv_kernel_sizes",
    "[5],[7,11],[9,15],[23]",
    "--residual_connection",
    "0,0,0,0",
    "--first_conv_filters",
    "32",
    "--first_conv_kernel_size",
    "5",
    "--stride",
    "3",
]
TFLITE = Path("tflite_stream_state_internal_quant") / "stream_state_internal_quant.tflite"


def model_args(profile: str) -> list[str]:
    args = list(MODEL_ARGS)
    if profile in {"current", "current_wider"}:
        values = {
            "--pointwise_filters": "40,40,40,40" if profile == "current_wider" else "32,32,32,32",
            "--first_conv_filters": "20" if profile == "current_wider" else "16",
            "--mixconv_kernel_sizes": "[5],[7],[9],[13]",
        }
        for key, value in values.items():
            args[args.index(key) + 1] = value
    return args


def _feature_set(path: Path, weight: float, truth: bool, strategy: str) -> dict:
    return {
        "features_dir": str(path),
        "sampling_weight": float(weight),
        "penalty_weight": 1.0,
        "truth": truth,
        "truncation_strategy": strategy,
        "type": "mmap",
    }


def feature_sets(project: Project, downloads: Path) -> list[dict]:
    weights = project.config["training"]["weights"]
    sets = []
    for name, truth, strategy in (
        ("own", True, "truncate_start"),
        ("own_hard", True, "truncate_start"),
        ("tts", True, "truncate_start"),
        ("own_negative", False, "random"),
        ("satellite_negative", False, "random"),
        ("speech_extra", False, "random"),
        ("background_negative", False, "random"),
        ("mined", False, "random"),
        ("tts_negative", False, "random"),
    ):
        if (project.features / name).is_dir():
            weight = weights.get(name, weights["own_negative"])
            sets.append(_feature_set(project.features / name, weight, truth, strategy))
    negatives = downloads / "negative_datasets"
    for name in project.config["negatives"]["feature_sets"]:
        if name == "dinner_party_eval":
            continue
        if (negatives / name).is_dir():
            sets.append(
                _feature_set(
                    negatives / name,
                    weights.get(name, weights.get(name.removesuffix("_background"), 5.0)),
                    False,
                    "random",
                )
            )
    if (negatives / "dinner_party_eval").is_dir():
        # Long background recordings: only for false activations per hour in validation and test.
        sets.append(_feature_set(negatives / "dinner_party_eval", 0.0, False, "split"))
    if not any(item["truth"] for item in sets):
        raise ProjectError("No positive features: run features first")
    if not any(not item["truth"] for item in sets):
        raise ProjectError("No negative features: run download (or generate negative phrases) first")
    return sets


def config(project: Project, downloads: Path) -> dict:
    settings = project.config["training"]
    return {
        **{
            key: settings[key]
            for key in (
                "device",
                "engine",
                "input_pipeline",
                "tfrecord_examples",
                "input_threads",
                "compute_threads",
                "prefetch_batches",
                "validation_cache_mb",
                "validation_batch_size",
                "progress_interval",
                "seed",
            )
        },
        "window_step_ms": 10,
        "train_dir": str(project.model_dir),
        "features": feature_sets(project, downloads),
        "training_steps": [int(settings["steps"])],
        "positive_class_weight": [float(settings["positive_class_weight"])],
        "negative_class_weight": [float(settings["negative_class_weight"])],
        "learning_rates": [float(settings["learning_rate"])],
        "early_stopping_patience": int(settings["early_stopping_patience"]),
        "early_stopping_min_delta": float(settings["early_stopping_min_delta"]),
        "lr_patience": int(settings["lr_patience"]),
        "lr_factor": float(settings["lr_factor"]),
        "lr_max_reductions": int(settings["lr_max_reductions"]),
        "min_learning_rate": float(settings["min_learning_rate"]),
        "batch_size": int(settings["batch_size"]),
        "time_mask_max_size": [int(settings["time_mask_max_size"])],
        "time_mask_count": [int(settings["time_mask_count"])],
        "freq_mask_max_size": [int(settings["freq_mask_max_size"])],
        "freq_mask_count": [int(settings["freq_mask_count"])],
        "eval_step_interval": int(settings["eval_step_interval"]),
        "clip_duration_ms": int(settings["clip_duration_ms"]),
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }


STEP_LINE = re.compile(r"Step #(\d+)")


def _stream(command: list[str], log_path: Path, on_step: Callable[[int], None] | None) -> int:
    """Run microWakeWord, echo and log its output, report training steps as they pass."""
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            env=environment(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
            match = STEP_LINE.search(line)
            if match and on_step:
                on_step(int(match.group(1)))
        return process.wait()


def train(
    project: Project,
    downloads: Path,
    resume: bool = True,
    runner=None,
    on_step: Callable[[int], None] | None = None,
) -> Path:
    if runner is None:
        from .datasets import check_features

        roots = [
            project.features / name
            for name in (
                "own",
                "own_hard",
                "tts",
                "own_negative",
                "satellite_negative",
                "speech_extra",
                "background_negative",
                "mined",
                "tts_negative",
            )
        ]
        roots += [downloads / "negative_datasets" / name for name in project.config["negatives"]["feature_sets"]]
        check_features(roots)
    if not resume:
        # microWakeWord creates the folder itself and refuses an existing one unless it resumes.
        shutil.rmtree(project.model_dir, ignore_errors=True)
    config_path = project.path("training_parameters.yaml")
    config_path.write_text(yaml.safe_dump(config(project, downloads), sort_keys=False), encoding="utf-8")
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        "-m",
        "wake_word_trainer.training_worker",
        "--runtime",
        str(config_path),
        f"--training_config={config_path}",
        "--train",
        "1",
        "--restore_checkpoint",
        "1" if resume else "0",
        "--test_tf_nonstreaming",
        "0",
        "--test_tflite_nonstreaming",
        "0",
        "--test_tflite_nonstreaming_quantized",
        "0",
        "--test_tflite_streaming",
        "0",
        "--test_tflite_streaming_quantized",
        "1",
        "--use_weights",
        "best_weights",
        *model_args(project.config["training"]["model_profile"]),
    ]
    seed = project.config["training"].get("seed")
    if project.config["training"]["early_stopping_patience"] or project.config["training"]["lr_patience"]:
        if project.config["training"]["engine"] == "standard":
            command[5:5] = ["--control"]
    if seed is not None:
        command[5:5] = ["--seed", str(seed)]
    model = project.model_dir / TFLITE
    model.unlink(missing_ok=True)  # never mistake the previous round's model for this one
    if runner is not None:
        returncode = runner(command, check=False).returncode
    else:
        returncode = _stream(command, project.path("training.log"), on_step)
    if returncode != 0 or not model.is_file():
        if returncode < 0:
            raise ProjectError(f"Training crashed ({signal.Signals(-returncode).name}); see training.log")
        raise ProjectError(f"Training failed (exit {returncode}); see training.log")
    return model
