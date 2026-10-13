"""Graph-Schritte mit gleichem Keras-Loss und vorbereiteter Eingabezuführung.

Übernimmt die Lab-Prinzipien (tf.function, TFRecord/Prefetch, Validierungs-Cache)
ohne Soft-Label-Loss oder die spätere Lab-Loss-Normalisierung zu aktivieren.
Modellbau, Validierungsmetriken und int8-Export bleiben beim gepinnten Backend.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .input_pipeline import batches
from .training_control import Controller
from .validation import StreamingValidation


def compile_model(model):
    import tensorflow as tf

    cutoffs = np.linspace(0, 1, 101).tolist()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.TruePositives(name="tp", thresholds=cutoffs),
            tf.keras.metrics.FalsePositives(name="fp", thresholds=cutoffs),
            tf.keras.metrics.TrueNegatives(name="tn", thresholds=cutoffs),
            tf.keras.metrics.FalseNegatives(name="fn", thresholds=cutoffs),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryCrossentropy(name="loss"),
        ],
    )


def spec_augment(x, config):
    import tensorflow as tf

    # Jede Zeile bekommt eigene Masken; dynamische Masken nach dem statischen Cache.
    shape = tf.shape(x)
    for axis, prefix in ((1, "time"), (2, "freq")):
        count, maximum = (
            int(config.get(prefix + "_mask_count", [0])[0]),
            int(config.get(prefix + "_mask_max_size", [0])[0]),
        )
        if not count or maximum <= 0:
            continue
        length = shape[axis]
        for _ in range(count):
            widths = tf.random.uniform((shape[0], 1), 0, tf.minimum(maximum, length) + 1, dtype=tf.int32)
            starts = tf.cast(tf.random.uniform((shape[0], 1)) * tf.cast(length - widths + 1, tf.float32), tf.int32)
            positions = tf.range(length)[None, :]
            mask = tf.logical_or(positions < starts, positions >= starts + widths)
            mask = mask[:, :, None] if axis == 1 else mask[:, None, :]
            x = x * tf.cast(mask, x.dtype)
    return x


def make_step(model, config, engine="optimized"):
    import tensorflow as tf
    from tensorflow.python.util import tf_decorator

    positive = float(config["positive_class_weight"][0])
    negative = float(config["negative_class_weight"][0])
    if engine == "standard":
        model.make_train_function()
        _, model.train_function = tf_decorator.unwrap(model.train_function)

        def standard(x, y, w):
            weights = w * tf.where(y >= 0.5, positive, negative)
            values = model.train_on_batch(x, y, sample_weight=weights)
            return tf.convert_to_tensor(values[0])

        return standard

    @tf.function(reduce_retracing=True, jit_compile=False)
    def graph(x, y, w):
        x = spec_augment(x, config)
        weights = w * tf.where(y >= 0.5, positive, negative)
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            # Keras berechnet wie bisher Summe der gewichteten Fehler / Batchgröße.
            loss = model.compute_loss(x=x, y=y, y_pred=predictions, sample_weight=weights, training=True)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables, strict=True))
        return loss

    return graph


def train(model, config, processor, backend):
    import tensorflow as tf

    compile_model(model)
    root = Path(config["train_dir"])
    checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
    checkpoint.restore(tf.train.latest_checkpoint(str(root / "restore")))
    iterator, input_stats = batches(processor, config)
    validation = StreamingValidation(config, processor, model)
    controller = Controller(
        early_patience=int(config.get("early_stopping_patience", 0)),
        lr_patience=int(config.get("lr_patience", 0)),
        factor=float(config.get("lr_factor", 0.5)),
        min_delta=float(config.get("early_stopping_min_delta", 0)),
        max_reductions=int(config.get("lr_max_reductions", 3)),
        min_lr=float(config.get("min_learning_rate", 1e-5)),
    )
    step = make_step(model, config)
    evaluations, timings, count = [], [], 0
    best_max, best_min = 0.0, 10000.0
    scheduled_rate = float(config["learning_rates"][0])
    steps = sum(config["training_steps"])
    started = time.monotonic()
    interval_started = started
    checkpoint_dir = root / "restore"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for number in range(1, steps + 1):
        model.optimizer.learning_rate.assign(scheduled_rate * controller.multiplier)
        begin = time.monotonic()
        batch = next(iterator)
        received = time.monotonic()
        loss = float(step(*batch).numpy())
        finished = time.monotonic()
        timings.append({"input_s": received - begin, "compute_s": finished - received})
        count = number
        if number % int(config.get("progress_interval", 25)) == 0:
            duration = finished - interval_started
            print(
                f"Step #{number}: loss={loss:.5f}, {config.get('progress_interval', 25) / duration:.2f} steps/s",
                flush=True,
            )
            interval_started = finished
        if number % int(config["eval_step_interval"]) == 0 or number == steps:
            begin = time.monotonic()
            model.save_weights(str(root / "last_weights.weights.h5"))
            metrics = validation("validation")
            model.reset_metrics()
            quality = float(metrics[config["maximization_metric"]])
            minimum = float(metrics[config["minimization_metric"]]) if config["minimization_metric"] else 0.0
            improved = (
                (
                    minimum <= config["target_minimization"]
                    and (quality > best_max or best_min > config["target_minimization"])
                )
                or (minimum > config["target_minimization"] and minimum < best_min)
                or (minimum == best_min and quality > best_max)
            )
            if improved:
                best_max, best_min = quality, minimum
                model.save_weights(str(root / "best_weights.weights.h5"))
                checkpoint.save(file_prefix=str(checkpoint_dir / "ckpt"))
            controller.observe(quality, scheduled_rate)
            evaluations.append(
                {
                    "step": number,
                    "metric": quality,
                    "loss": float(metrics["loss"]),
                    "seconds": time.monotonic() - begin,
                    "lr_multiplier": controller.multiplier,
                    "stop": controller.stop,
                    "cache_bytes": validation.cache_bytes,
                }
            )
            print(f"Validation at step {number}: quality={quality:.6f}, {evaluations[-1]['seconds']:.2f}s", flush=True)
            interval_started = time.monotonic()
            if controller.stop:
                print(f"Early stopping at step {number}", flush=True)
                break
    checkpoint.save(file_prefix=str(checkpoint_dir / "ckpt"))
    model.save_weights(str(root / "last_weights.weights.h5"))
    (root / "training_control.json").write_text(json.dumps(evaluations, indent=2))
    (root / "performance.json").write_text(
        json.dumps(
            {
                "engine": "optimized",
                "input": input_stats,
                "steps": count,
                "seconds": time.monotonic() - started,
                "input_seconds": sum(row["input_s"] for row in timings),
                "compute_seconds": sum(row["compute_s"] for row in timings),
                "evaluations": evaluations,
            },
            indent=2,
        )
    )


def install(backend):
    backend.train = lambda model, config, processor: train(model, config, processor, backend)
