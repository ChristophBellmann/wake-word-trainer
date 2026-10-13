"""Der optimierte Kern muss Frühstopp, Checkpoints und int8-Export erhalten."""

import json

import pytest


@pytest.mark.slow
def test_optimized_training_stops_and_exports(tmp_path):
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
            "device": "cpu",
            "engine": "optimized",
            "input_pipeline": "tfrecord",
            "tfrecord_examples": 128,
            "model_profile": "current",
            "steps": 50,
            "eval_step_interval": 10,
            "early_stopping_patience": 1,
            "lr_patience": 1,
            "batch_size": 4,
            "seed": 42,
            "time_mask_max_size": 2,
            "time_mask_count": 1,
            "freq_mask_max_size": 2,
            "freq_mask_count": 1,
        },
    )
    for split in ("training", "validation", "testing"):
        for name, clips in [
            ("own", [chirp(seed=s) for s in range(4)]),
            ("own_negative", [noise(0.8, s) for s in range(4)]),
        ]:
            write_mmap(
                project.features / name / split / f"{name}_mmap", (generate_features_for_clip(c, 10) for c in clips)
            )
    model = train(project, tmp_path / "downloads", resume=False)
    history = json.loads(project.path("model/training_control.json").read_text())
    assert history[-1]["step"] == 20 and history[-1]["stop"]
    assert model.stat().st_size > 1000 and project.path("model/best_weights.weights.h5").is_file()
    assert json.loads(project.path("training_runtime.json").read_text())["device"] == "/CPU:0"
