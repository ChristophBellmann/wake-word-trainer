from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from wake_word_trainer import audio
from wake_word_trainer.features import parallel_spectrograms, spectrograms
from wake_word_trainer.gpu import busy, lease
from wake_word_trainer.project import Project


@pytest.mark.slow
def test_spawned_feature_workers_preserve_all_frames_and_order(tmp_path):
    project = Project.create(tmp_path / "project", "Test")
    project.config["training"]["feature_workers"] = 2
    clips = []
    for index in range(3):
        path = tmp_path / f"{index}.wav"
        audio.write(path, np.random.default_rng(index).normal(0, 0.03, 24000).astype(np.float32))
        clips.append(path)
    expected = list(spectrograms(clips, None, 2, 3))
    actual = list(parallel_spectrograms(project, tmp_path, clips, 2, 3, None))
    assert len(actual) == len(expected) == 18
    for first, second in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(first, second)


def test_child_uses_parent_lease_without_releasing_it(tmp_path):
    path = tmp_path / "gpu.lock"
    code = (
        "from pathlib import Path; from wake_word_trainer.gpu import lease; "
        "import sys; "
        "guard=lease('child',Path(sys.argv[1]),int(sys.argv[2])); "
        "guard.__enter__(); guard.__exit__(None,None,None)"
    )
    with lease("parent", path) as handle:
        subprocess.run([sys.executable, "-c", code, str(path), str(handle.fileno())],
                       pass_fds=(handle.fileno(),), env=dict(os.environ), check=True)
        assert busy(path)
    assert not busy(path)
