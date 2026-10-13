"""Begrenzte tf.data-Zuführung und gestempelter TFRecord-Pool aus dem Lab-Prinzip.

Keine RAM-Kopie aller MMAPs. Statische Features werden ohne SpecAugment
gespeichert; Zeit-/Frequenzmasken werden bei jedem Trainingsschritt neu gezogen.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


def signature(config: dict, count: int) -> dict:
    files = []
    for source in config["features"]:
        for path in sorted(Path(source["features_dir"]).rglob("*.ninja")):
            stat = path.stat()
            files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return {
        "version": 1,
        "examples": count,
        "shape": list(config["training_input_shape"]),
        "length": config["spectrogram_length"],
        "seed": config.get("seed"),
        "sources": config["features"],
        "files": hashlib.sha256(json.dumps(files).encode()).hexdigest(),
    }


def raw_batch(processor, config, size):
    x, y, w = processor.get_data("training", size, config["spectrogram_length"], "default")
    return np.asarray(x, np.float32), np.asarray(y, np.float32).reshape(-1, 1), np.asarray(w, np.float32).reshape(-1, 1)


def tfrecord(processor, config: dict, examples: int) -> tuple[Path, dict]:
    import tensorflow as tf

    stamp = signature(config, examples)
    key = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    root = Path(config.get("input_cache_dir", str(Path(config["train_dir"]).parent / "input_cache")))
    root.mkdir(parents=True, exist_ok=True)
    path, metadata = root / f"{key}.tfrecord", root / f"{key}.json"
    start = time.monotonic()
    if path.is_file() and metadata.is_file() and json.loads(metadata.read_text()) == stamp:
        return path, {"reused": True, "examples": examples, "seconds": time.monotonic() - start, "signature": key}
    temporary = path.with_suffix(".part")
    with tf.io.TFRecordWriter(str(temporary)) as writer:
        written = 0
        while written < examples:
            size = min(512, examples - written)
            xs, ys, ws = raw_batch(processor, config, size)
            for x, y, w in zip(xs, ys, ws, strict=True):
                row = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            # Float32 erhält exakt dieselben Werte wie der bisherige Pfad.
                            "x": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.tobytes()])),
                            "y": tf.train.Feature(float_list=tf.train.FloatList(value=[float(y[0])])),
                            "w": tf.train.Feature(float_list=tf.train.FloatList(value=[float(w[0])])),
                        }
                    )
                )
                writer.write(row.SerializeToString())
            written += size
            if written % 8192 == 0 or written == examples:
                print(f"Input cache: {written}/{examples} examples", flush=True)
    temporary.replace(path)
    pending = metadata.with_suffix(".part")
    pending.write_text(json.dumps(stamp, sort_keys=True))
    pending.replace(metadata)
    return path, {"reused": False, "examples": examples, "seconds": time.monotonic() - start, "signature": key}


def batches(processor, config: dict) -> tuple[object, dict]:
    import tensorflow as tf

    size = int(config["batch_size"])
    shape = tuple(config["training_input_shape"])
    mode = config.get("input_pipeline", "prefetch")
    meta = {"pipeline": mode}
    if mode == "python":

        def generator():
            while True:
                yield raw_batch(processor, config, size)

        return iter(generator()), meta
    if mode == "tfrecord":
        # Kurzläufe erzeugen keinen riesigen Pool für wenige Hundert Beispiele.
        count = min(int(config.get("tfrecord_examples", 65536)), max(1024, size * sum(config["training_steps"])))
        path, stats = tfrecord(processor, config, count)
        meta.update(stats)
        fields = {
            "x": tf.io.FixedLenFeature([], tf.string),
            "y": tf.io.FixedLenFeature([], tf.float32),
            "w": tf.io.FixedLenFeature([], tf.float32),
        }

        def parse(value):
            item = tf.io.parse_single_example(value, fields)
            return (
                tf.reshape(tf.io.decode_raw(item["x"], tf.float32), shape),
                tf.reshape(item["y"], (1,)),
                tf.reshape(item["w"], (1,)),
            )

        dataset = tf.data.TFRecordDataset(str(path)).shuffle(
            min(count, 8192), seed=config.get("seed"), reshuffle_each_iteration=True
        )
        dataset = dataset.repeat().map(parse, num_parallel_calls=int(config.get("input_threads", 4)))
        dataset = dataset.batch(size, drop_remainder=True)
    else:

        def generator():
            while True:
                yield raw_batch(processor, config, size)

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec((size, *shape), tf.float32),
                tf.TensorSpec((size, 1), tf.float32),
                tf.TensorSpec((size, 1), tf.float32),
            ),
        )
    options = tf.data.Options()
    options.threading.private_threadpool_size = int(config.get("input_threads", 4))
    options.threading.max_intra_op_parallelism = 1
    options.experimental_deterministic = True
    dataset = dataset.with_options(options).prefetch(int(config.get("prefetch_batches", 2)))
    return iter(dataset), meta


class ValidationCache:
    """Nur passende, begrenzte Holdout-Arrays halten; bei RAM-Druck verwerfen."""

    def __init__(self, processor, megabytes: int = 512):
        self.processor = processor
        self.limit = megabytes * 1024**2
        self.cached = {}
        self.used = 0

    def get_data(self, mode, *args, **kwargs):
        if mode == "training":
            return self.processor.get_data(mode, *args, **kwargs)
        from .system_check import memory

        key = (mode, repr(args), repr(sorted(kwargs.items())))
        if memory().get("available_memory_gb", 0) < 6:
            self.cached.clear()
            self.used = 0
        if key in self.cached:
            return self.cached[key]
        rows = self.processor.get_data(mode, *args, **kwargs)
        required = sum(getattr(item, "nbytes", 0) for item in rows)
        if self.limit and self.used + required <= self.limit and memory().get("available_memory_gb", 0) >= 6:
            self.cached[key] = rows
            self.used += required
        return rows

    def __getattr__(self, name):
        return getattr(self.processor, name)
