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


def test_native_worker_crash_replays_every_pending_clip_in_order(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool

    from wake_word_trainer import features

    submitted, generations = [], []

    class Pool:
        def __init__(self, **kwargs):
            generations.append(self)
            self.generation = len(generations)

        def submit(self, function, job):
            submitted.append((self.generation, job[0]))
            future = Future()
            if self.generation == 1:
                future.set_exception(BrokenProcessPool("simulierter nativer Absturz"))
            else:
                future.set_result([np.array([[int(Path(job[0]).stem)]])])
            return future

        def shutdown(self, **kwargs):
            pass

    from pathlib import Path

    monkeypatch.setattr(features, "ProcessPoolExecutor", Pool)
    project = Project.create(tmp_path / "project", "Test")
    project.config["training"]["feature_workers"] = 2
    clips = [tmp_path / f"{i}.wav" for i in range(8)]
    result = list(parallel_spectrograms(project, tmp_path, clips, 1, 1, None))
    assert [int(x[0, 0]) for x in result] == list(range(8))
    assert len(generations) == 2
    assert all((2, str(path)) in submitted for path in clips)


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


@pytest.mark.slow
def test_partial_mmap_is_rejected_until_rebuild_finishes(tmp_path):
    from wake_word_trainer.datasets import check_features
    from wake_word_trainer.features import write_mmap
    from wake_word_trainer.project import ProjectError

    target = tmp_path / "features" / "own_mmap"

    def interrupted():
        for _ in range(201):
            yield np.ones((5, 40), np.float32)
        raise RuntimeError("unterbrochener Worker")

    with pytest.raises(RuntimeError, match="unterbrochener Worker"):
        write_mmap(target, interrupted())
    with pytest.raises(ProjectError, match="Unvollständiger Feature-Bau"):
        check_features([target.parent])
    write_mmap(target, (np.ones((5, 40), np.float32) for _ in range(201)))
    assert check_features([target.parent]) == 1
