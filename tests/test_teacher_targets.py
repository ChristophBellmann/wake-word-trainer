"""Separate Teacher targets retain short positives without modifying Student data."""

import numpy as np
import pytest

from wake_word_trainer import audio
from wake_word_trainer.project import Project, ProjectError
from wake_word_trainer.teacher import source_balance, training_sources


def test_short_teacher_only_positive_is_kept_and_has_own_sampling_mass(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova", tts={"min_positive_duration_seconds": 0.6})
    extra = tmp_path / "teacher_only"
    audio.write(extra / "short.wav", np.ones(4000, np.float32) * 0.03)
    audio.write(project.path("tts/positive/short.wav"), np.ones(4000, np.float32) * 0.08)
    audio.write(project.recordings / "train/room/positive.wav", np.ones(16000, np.float32) * 0.1)
    audio.write(project.recordings / "negative/train/room/negative.wav", np.ones(16000, np.float32) * 0.01)
    sources = training_sources(project, True, extra_positive=[extra])
    assert extra / "short.wav" in sources["train_positive"]
    assert project.path("tts/positive/short.wav") not in sources["train_positive"]
    weights, _, counts = source_balance(project, sources, [extra], 3)
    assert counts["teacher_extra_positive"] == 1
    assert weights[0] == pytest.approx(3) and weights[1] == pytest.approx(3)
    assert len(audio.wavs(project.recordings / "train")) == 1


def test_teacher_extra_holdout_never_enters_optimizer(tmp_path):
    project = Project.create(tmp_path / "p", "Hey Nova")
    train = tmp_path / "extra_train"
    evaluation = tmp_path / "extra_eval"
    samples = np.ones(16000, np.float32) * 0.1
    audio.write(train / "same.wav", samples)
    audio.write(evaluation / "same.wav", samples)
    with pytest.raises(ProjectError, match="verschiedenen Datenrollen"):
        training_sources(project, extra_positive=[train], extra_eval_positive=[evaluation])
