"""Frühstopp folgt der Bewertung erst nach dem Sichern der besten Gewichte."""

import json
import sys
from types import SimpleNamespace

import pytest

from wake_word_trainer.training_control import Controller, install


def test_plateau_halves_learning_rate_and_stops_without_resetting_patience():
    control = Controller(early_patience=6, lr_patience=2)
    assert not control.observe(0.8)
    reductions = [control.observe(0.8) for _ in range(6)]
    assert reductions == [False, True, False, True, False, True]
    assert control.multiplier == 0.125 and control.stop
    improved = Controller(early_patience=3, lr_patience=2)
    for value in [0.4, 0.4, 0.5, 0.5]:
        improved.observe(value)
    assert improved.stale == 1 and not improved.stop


def test_backend_finishes_checkpoint_selection_before_stopping(tmp_path, monkeypatch):
    events = []

    class Rate:
        def __init__(self):
            self.value = 0.001

        def assign(self, value):
            self.value = value

        def numpy(self):
            return self.value

    rate = Rate()
    model = SimpleNamespace(
        optimizer=SimpleNamespace(learning_rate=rate),
        train_on_batch=lambda: events.append(("batch", rate.value)),
        save_weights=lambda path: events.append(("save", path)),
    )
    original_batch = model.train_on_batch
    backend = SimpleNamespace()

    def validate(*a):
        return {"quality": 0.8}

    backend.validate_nonstreaming = validate

    def train(model, config, data_processor):
        for step in range(10):
            rate.assign(0.001)
            model.train_on_batch()
            backend.validate_nonstreaming(config, data_processor, model, "validation")
            events.append(("best_selection", step))

    backend.train = train
    monkeypatch.setitem(
        sys.modules,
        "tensorflow",
        SimpleNamespace(
            train=SimpleNamespace(
                Checkpoint=lambda **kw: SimpleNamespace(save=lambda **kw: events.append(("checkpoint", kw)))
            )
        ),
    )
    install(backend)
    backend.train(
        model,
        {"train_dir": str(tmp_path), "maximization_metric": "quality", "early_stopping_patience": 3, "lr_patience": 1},
        None,
    )
    assert [event[1] for event in events if event[0] == "batch"] == pytest.approx([0.001, 0.001, 0.0005, 0.00025])
    assert len([event for event in events if event[0] == "best_selection"]) == 4
    assert events[-2][0] == "checkpoint" and events[-1][0] == "save"
    assert backend.validate_nonstreaming is validate and model.train_on_batch is original_batch
    history = json.loads((tmp_path / "training_control.json").read_text())
    assert history[-1]["step"] == 4 and history[-1]["stop"]


@pytest.mark.slow
def test_real_backend_early_stop_still_exports_streaming_model(tmp_path):
    pytest.importorskip("microwakeword")
    from microwakeword.audio.audio_utils import generate_features_for_clip

    from wake_word_trainer.features import write_mmap
    from wake_word_trainer.project import Project
    from wake_word_trainer.training import train

    from .conftest import chirp, noise

    project = Project.create(
        tmp_path / "p",
        "Hey Nova",
        training={
            "steps": 50,
            "eval_step_interval": 10,
            "batch_size": 4,
            "model_profile": "current",
            "early_stopping_patience": 1,
            "lr_patience": 1,
            "seed": 42,
        },
    )
    for split in ("training", "validation", "testing"):
        for name, clips in (
            ("own", [chirp(seed=s) for s in range(4)]),
            ("own_negative", [noise(0.8, s) for s in range(4)]),
        ):
            write_mmap(
                project.features / name / split / f"{name}_mmap",
                (generate_features_for_clip(clip, 10) for clip in clips),
            )
    model = train(project, tmp_path / "downloads", resume=False)
    history = json.loads(project.path("model/training_control.json").read_text())
    # Ohne Ambient-Set ist average_viable_recall konstant null: echte Plateau-Prüfung.
    assert history[-1]["step"] == 20 and history[-1]["stop"]
    assert project.path("model/best_weights.weights.h5").is_file()
    assert model.stat().st_size > 1000
