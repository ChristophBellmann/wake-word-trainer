"""Training with microWakeWord (MixedNet, streaming, int8 quantized)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

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
        ("tts", True, "truncate_start"),
        ("tts_negative", False, "truncate_start"),
    ):
        if (project.features / name).is_dir():
            sets.append(_feature_set(project.features / name, weights[name], truth, strategy))
    negatives = downloads / "negative_datasets"
    for name in ("speech", "dinner_party", "no_speech"):
        if (negatives / name).is_dir():
            sets.append(_feature_set(negatives / name, weights[name], False, "random"))
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
        "window_step_ms": 10,
        "train_dir": str(project.model_dir),
        "features": feature_sets(project, downloads),
        "training_steps": [int(settings["steps"])],
        "positive_class_weight": [float(settings["positive_class_weight"])],
        "negative_class_weight": [float(settings["negative_class_weight"])],
        "learning_rates": [float(settings["learning_rate"])],
        "batch_size": int(settings["batch_size"]),
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": int(settings["eval_step_interval"]),
        "clip_duration_ms": int(settings["clip_duration_ms"]),
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }


def train(project: Project, downloads: Path, resume: bool = True, runner=subprocess.run) -> Path:
    project.model_dir.mkdir(parents=True, exist_ok=True)
    config_path = project.model_dir / "training_parameters.yaml"
    config_path.write_text(yaml.safe_dump(config(project, downloads), sort_keys=False), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "microwakeword.model_train_eval",
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
        *MODEL_ARGS,
    ]
    result = runner(command, check=False)
    model = project.model_dir / TFLITE
    if result.returncode != 0 or not model.is_file():
        raise ProjectError(f"Training failed (exit {result.returncode}); see the output above")
    return model
