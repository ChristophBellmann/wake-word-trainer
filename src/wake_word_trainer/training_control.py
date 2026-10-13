"""Optionaler LR-Abfall und Frühstopp nach vollständigen Backend-Bewertungen."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .project import ProjectError


@dataclass
class Controller:
    early_patience: int = 0
    lr_patience: int = 0
    factor: float = 0.5
    min_delta: float = 0.0
    max_reductions: int = 3
    min_lr: float = 1e-5
    reductions: int = 0
    best: float = -math.inf
    stale: int = 0
    lr_stale: int = 0
    multiplier: float = 1.0
    stop: bool = False

    def observe(self, value: float, scheduled_rate: float = 0.001) -> bool:
        if not math.isfinite(value):
            raise ProjectError("Vollständige Bewertung lieferte einen ungültigen Trainingswert")
        if value > self.best + self.min_delta:
            self.best, self.stale, self.lr_stale = value, 0, 0
        else:
            self.stale += 1
            self.lr_stale += 1
        reduced = (
            self.lr_patience > 0
            and self.lr_stale >= self.lr_patience
            and self.reductions < self.max_reductions
            and scheduled_rate * self.multiplier * self.factor >= self.min_lr
        )
        if reduced:
            self.multiplier *= self.factor
            self.reductions += 1
            self.lr_stale = 0
        self.stop = self.early_patience > 0 and self.stale >= self.early_patience
        return reduced


class EarlyStop(Exception):
    pass


def install(backend) -> None:
    """Hook nur im frischen Trainingsprozess; Backend und Modell werden zurückgesetzt.

    Der gepinnte Backend prüft bei jedem eval_step_interval den gesamten Holdout.
    Ein Frühstopp erfolgt erst beim nächsten Batch, nachdem er die besten Gewichte
    und den Checkpoint der vorausgehenden Bewertung gesichert hat.
    """
    original_train, original_validate = backend.train, backend.validate_nonstreaming

    def controlled(model, config, data_processor):
        controller = Controller(
            early_patience=int(config.get("early_stopping_patience", 0)),
            lr_patience=int(config.get("lr_patience", 0)),
            factor=float(config.get("lr_factor", 0.5)),
            min_delta=float(config.get("early_stopping_min_delta", 0)),
            max_reductions=int(config.get("lr_max_reductions", 3)),
            min_lr=float(config.get("min_learning_rate", 1e-5)),
        )
        if not (controller.early_patience or controller.lr_patience):
            return original_train(model, config, data_processor)
        batch = model.train_on_batch
        step, history, scheduled_rate = 0, [], 0.001

        def train_batch(*args, **kwargs):
            nonlocal step, scheduled_rate
            if controller.stop:
                raise EarlyStop
            rate = model.optimizer.learning_rate
            scheduled_rate = float(rate.numpy())
            rate.assign(scheduled_rate * controller.multiplier)
            step += 1
            return batch(*args, **kwargs)

        def validate(*args, **kwargs):
            metrics = original_validate(*args, **kwargs)
            value = float(metrics[config["maximization_metric"]])
            reduced = controller.observe(value, scheduled_rate)
            history.append(
                {
                    "step": step,
                    "metric": value,
                    "lr_multiplier": controller.multiplier,
                    "stale": controller.stale,
                    "stop": controller.stop,
                }
            )
            if reduced:
                print(f"Learning rate multiplier: {controller.multiplier:.5f} at step {step}", flush=True)
            return metrics

        model.train_on_batch, backend.validate_nonstreaming = train_batch, validate
        try:
            original_train(model, config, data_processor)
        except EarlyStop:
            import tensorflow as tf

            print(f"Early stopping at step {step} after {controller.stale} evaluations without improvement", flush=True)
            checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
            checkpoint.save(file_prefix=str(Path(config["train_dir"]) / "restore" / "ckpt"))
            model.save_weights(str(Path(config["train_dir"]) / "last_weights.weights.h5"))
        finally:
            model.train_on_batch, backend.validate_nonstreaming = batch, original_validate
            Path(config["train_dir"], "training_control.json").write_text(json.dumps(history, indent=2))

    backend.train = controlled
