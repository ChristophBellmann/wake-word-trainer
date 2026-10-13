"""Vollständige Validierung in begrenzten Batches, mit statischem Cache pro Lauf.

Alle Provider und alle Ambient-Fenster bleiben erhalten. Keine kompletten
mehrgigabytegroßen NumPy-Arrays vor jedem Sweep aufbauen.
"""

from __future__ import annotations

import numpy as np


class StreamingValidation:
    def __init__(self, config, processor, model):
        import tensorflow as tf

        self.config, self.processor, self.model = config, processor, model
        self.datasets = {}
        self.cache_bytes = 0
        self.limit = int(config.get("validation_cache_mb", 512)) * 1024**2
        self.batch_size = int(config.get("validation_batch_size", 1024))

        @tf.function(reduce_retracing=True, jit_compile=False)
        def step(x, y):
            return model.test_step((x, y))

        self.step = step

    def dataset(self, mode, strategy):
        import tensorflow as tf

        key = mode, strategy
        if key in self.datasets:
            return self.datasets[key]
        shape = tuple(self.config["training_input_shape"])
        length = self.config["spectrogram_length"]
        providers = [p for p in self.processor.feature_providers if p.get_mode_size(mode)]

        def generator():
            for provider in providers:
                for x in provider.get_feature_generator(mode, length, strategy):
                    yield np.asarray(x, np.float32), np.array([provider.label], np.float32)

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape, tf.float32),
                tf.TensorSpec((1,), tf.float32),
            ),
        ).batch(self.batch_size)
        if strategy == "split":
            # Derselbe 1000*step*stride-Frame-Abstand wie im gepinnten Provider.
            frames = sum(p.get_mode_duration(mode) for p in providers) / (self.config["window_step_ms"] / 1000)
            examples = frames / (self.config["window_step_ms"] * self.config["stride"])
        else:
            examples = sum(p.get_mode_size(mode) * len(getattr(p, "fixed_right_cutoffs", [0])) for p in providers)
        estimated = int(examples * (np.prod(shape) * 4 + 4) * 1.05)
        from .system_check import memory

        headroom = max(0, memory().get("available_memory_gb", 0) - 6) * 1024**3 / 2
        if self.limit and self.cache_bytes + estimated <= min(self.limit, headroom):
            dataset = dataset.cache()
            self.cache_bytes += estimated
        options = tf.data.Options()
        options.threading.private_threadpool_size = int(self.config.get("input_threads", 4))
        options.threading.max_intra_op_parallelism = 1
        dataset = dataset.with_options(options).prefetch(int(self.config.get("prefetch_batches", 2)))
        self.datasets[key] = dataset
        return dataset

    def evaluate(self, mode, strategy):
        for x, y in self.dataset(mode, strategy):
            self.step(x, y)
        result = self.model.get_metrics_result()
        return {name: value.numpy() for name, value in result.items()}

    def __call__(self, mode="validation"):
        self.model.reset_metrics()
        ordinary = self.evaluate(mode, "truncate_start")
        metrics = {key: float(ordinary[key]) for key in ("accuracy", "recall", "precision", "auc", "loss")}
        metrics.update(
            recall_at_no_faph=0.0,
            cutoff_for_no_faph=0.0,
            ambient_false_positives=0.0,
            ambient_false_positives_per_hour=0.0,
            average_viable_recall=0.0,
        )
        if not self.processor.get_mode_size(mode + "_ambient"):
            return metrics
        combined = self.evaluate(mode + "_ambient", "split")
        fp = np.asarray(combined["fp"]) - np.asarray(ordinary["fp"])
        tp, fn = np.asarray(combined["tp"]), np.asarray(combined["fn"])
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=tp + fn > 0)
        faph = fp / (self.processor.get_mode_duration(mode + "_ambient") / 3600)
        cutoffs = np.linspace(0, 1, 101)
        first = np.flatnonzero(faph == 0)[0]
        no_faph_recall = float(recall[first])
        if faph[0] > 2:
            index = np.flatnonzero(faph <= 2)[0]
            x0, y0 = faph[index - 1], recall[index - 1]
            x1, y1 = faph[index], recall[index]
            recall_at_two = float((y0 * (x1 - 2) + y1 * (2 - x0)) / (x1 - x0))
        else:
            recall_at_two = float(recall[0])
        # Identische Budgetfläche wie der gepinnte Backend, inklusive Nullpunkt.
        xs, ys = [2.0], [recall_at_two]
        for index, value in enumerate(faph):
            if value < 2 and value != xs[-1]:
                xs.append(float(value))
                ys.append(float(recall[index]))
        if xs[-1] != 0:
            xs.append(0.0)
            ys.append(no_faph_recall)
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        area = float(trapezoid(np.flip(ys), np.flip(xs)) / 2)
        metrics.update(
            auc=float(combined["auc"]),
            loss=float(combined["loss"]),
            recall_at_no_faph=no_faph_recall,
            cutoff_for_no_faph=float(cutoffs[first]),
            ambient_false_positives=float(fp[50]),
            ambient_false_positives_per_hour=float(faph[50]),
            average_viable_recall=area,
        )
        return metrics
