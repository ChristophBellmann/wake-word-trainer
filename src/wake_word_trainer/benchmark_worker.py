"""Frischer Framework-Prozess pro Benchmark; beendet vor Export/Modellübernahme."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from .gpu import lease, lock_path, snapshot, tensorflow_device
from .input_pipeline import ValidationCache, batches, signature
from .optimized_tf import compile_model, make_step
from .validation import StreamingValidation


class Complete(Exception):
    pass


def measure(model, config, processor, backend, args, runtime):
    compile_model(model)
    model.optimizer.learning_rate.assign(float(config["learning_rates"][0]))
    step = make_step(model, config, config["engine"])
    started = time.monotonic()
    iterator, stats = batches(processor, config)
    input_seconds = time.monotonic() - started
    warmup_start = time.monotonic()
    for _ in range(args.warmup):
        value = step(*next(iterator))
        value.numpy()
    warmup_seconds = time.monotonic() - warmup_start
    samples = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            samples.append(snapshot())
            stop.wait(0.2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    begin = time.monotonic()
    losses = []
    waiting = 0.0
    computing = 0.0
    try:
        for _ in range(args.steps):
            before = time.monotonic()
            batch = next(iterator)
            ready = time.monotonic()
            loss = step(*batch)
            losses.append(float(loss.numpy()))  # GPU synchronisieren, keine bloße Kernel-Startzeit messen.
            end = time.monotonic()
            waiting += ready - before
            computing += end - ready
    finally:
        stop.set()
        thread.join(timeout=2)
    elapsed = time.monotonic() - begin
    cache = ValidationCache(processor, int(config.get("validation_cache_mb", 512)))
    streamed = StreamingValidation(config, processor, model)
    validation = []
    for pass_number in range(2):
        start = time.monotonic()
        metrics = (
            streamed()
            if config["engine"] == "optimized"
            else backend.validate_nonstreaming(config, cache, model, "validation")
        )
        validation.append(
            {
                "pass": pass_number + 1,
                "seconds": time.monotonic() - start,
                "loss": float(metrics["loss"]),
                "quality": float(metrics["average_viable_recall"]),
                "cache_bytes": streamed.cache_bytes if config["engine"] == "optimized" else cache.used,
            }
        )
    valid = bool(np.isfinite(losses).all()) and all(np.isfinite(r["loss"]) for r in validation)
    result = {
        "valid": valid,
        "runtime": runtime,
        "feature_signature": signature(config, stats.get("examples", 0)),
        "input": stats,
        "input_preparation_seconds": input_seconds,
        "warmup_seconds": warmup_seconds,
        "measured_seconds": elapsed,
        "input_wait_seconds": waiting,
        "compute_seconds": computing,
        "steps_per_second": args.steps / elapsed,
        "examples_per_second": args.steps * config["batch_size"] / elapsed,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "training_loss_kind": "unweighted_running_metric"
        if config["engine"] == "standard"
        else "weighted_batch_objective",
        "validation": validation,
        "gpu": {
            "mean_use_percent": float(np.mean([r["use_percent"] for r in samples if "use_percent" in r]))
            if samples and any("use_percent" in r for r in samples)
            else 0,
            "peak_vram_bytes": max((r.get("vram_used_bytes", 0) for r in samples), default=0),
            "mean_power_watts": float(np.mean([r["power_watts"] for r in samples if "power_watts" in r]))
            if any("power_watts" in r for r in samples)
            else None,
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    raise Complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--gpu-lock-fd", type=int)
    args, remaining = parser.parse_known_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.gpu_lock_fd is not None:
        inherited = os.fstat(args.gpu_lock_fd)
        expected = lock_path().stat()
        if (inherited.st_ino, inherited.st_dev) != (expected.st_ino, expected.st_dev):
            raise ValueError("Ungültige geerbte GPU-Sperre")
    guard = contextlib.nullcontext() if args.gpu_lock_fd is not None else lease("gpu-benchmark")
    with guard:
        runtime = tensorflow_device(config["device"], int(config.get("compute_threads", 4)))
        import microwakeword.train as backend
        import tensorflow as tf

        tf.keras.utils.set_random_seed(42)
        backend.train = lambda model, cfg, processor: measure(model, cfg, processor, backend, args, runtime)
        sys.argv = [
            "microwakeword.model_train_eval",
            f"--training_config={args.config}",
            "--train",
            "1",
            "--restore_checkpoint",
            "0",
            "--test_tf_nonstreaming",
            "0",
            "--test_tflite_nonstreaming",
            "0",
            "--test_tflite_nonstreaming_quantized",
            "0",
            "--test_tflite_streaming",
            "0",
            "--test_tflite_streaming_quantized",
            "0",
            *remaining,
        ]
        try:
            runpy.run_module("microwakeword.model_train_eval", run_name="__main__")
        except Complete:
            pass


if __name__ == "__main__":
    main()
