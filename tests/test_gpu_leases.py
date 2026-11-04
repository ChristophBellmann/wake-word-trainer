"""Ownership regression fixtures; CPU/stdlib only, no framework initialization."""

import asyncio
import contextvars
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from wake_word_trainer.gpu import busy, lease
from wake_word_trainer.project import ProjectError


def test_nested_audit_teacher_keeps_one_owner(tmp_path):
    path = tmp_path / "gpu.lock"
    with lease("STT-Datenprüfung", path) as outer:
        metadata = path.read_bytes()
        for label in ("teacher-initialize", "teacher-scoring"):
            with lease(label, path) as inner:
                assert inner is outer
                assert busy(path)
                assert path.read_bytes() == metadata
        assert busy(path)
    assert not busy(path)
    assert path.read_bytes() == b""


def test_audit_gpu_context_can_enclose_teacher_operations(tmp_path, monkeypatch):
    from wake_word_trainer import audit

    path = tmp_path / "gpu.lock"
    monkeypatch.setattr("wake_word_trainer.gpu.lock_path", lambda: path)

    def nested_operations(*args):
        for label in ("teacher-initialize", "teacher-scoring"):
            with lease(label):
                assert busy(path)
                assert json.loads(path.read_text())["job"] == "STT-Datenprüfung"
        return {"fixture": True}

    monkeypatch.setattr(audit, "_run", nested_operations)
    assert audit.run(None, [], tmp_path / "unused.json", stt_device="gpu", stt_backend="torch_whisper") == {
        "fixture": True
    }
    assert not busy(path)


def test_nested_exception_does_not_release_outer(tmp_path):
    path = tmp_path / "gpu.lock"
    with pytest.raises(RuntimeError, match="outer"), lease("outer", path):
        with pytest.raises(ValueError, match="inner"), lease("inner", path):
            raise ValueError("inner")
        assert busy(path)
        assert json.loads(path.read_text())["job"] == "outer"
        raise RuntimeError("outer")
    assert not busy(path)
    with lease("next", path):
        assert busy(path)


@pytest.mark.parametrize("explicit_fd", [False, True])
def test_other_thread_cannot_reuse_even_with_copied_context(tmp_path, explicit_fd):
    path = tmp_path / "gpu.lock"
    with lease("owner", path) as handle:
        copied = contextvars.copy_context()

        def contender():
            with (
                pytest.raises(ProjectError, match="bereits"),
                lease("other-thread", path, inherited_fd=handle.fileno() if explicit_fd else None),
            ):
                pytest.fail("foreign thread entered")

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(copied.run, contender).result(timeout=10)
        assert busy(path)


def test_child_task_cannot_reuse_inherited_context(tmp_path):
    path = tmp_path / "gpu.lock"

    async def contender():
        with pytest.raises(ProjectError, match="bereits"), lease("child-task", path):
            pytest.fail("foreign task entered")

    async def owner():
        with lease("owner-task", path):
            with lease("lexical", path):
                assert busy(path)
            await asyncio.create_task(contender())
            assert busy(path)

    asyncio.run(owner())
    assert not busy(path)


def test_saved_context_has_no_rights_after_release(tmp_path):
    path = tmp_path / "gpu.lock"
    with lease("old", path):
        saved = contextvars.copy_context()

    def stale():
        with pytest.raises(ProjectError, match="bereits"), lease("stale", path):
            pytest.fail("expired context reused")

    with lease("current", path):
        saved.run(stale)
        assert busy(path)


def test_distinct_lock_paths_get_distinct_owners(tmp_path):
    first, second = tmp_path / "one.lock", tmp_path / "two.lock"
    with lease("one", first) as one:
        with lease("two", second) as two:
            assert one is not two
            assert busy(first) and busy(second)
        assert busy(first) and not busy(second)


def test_replaced_lockfile_is_not_reused(tmp_path):
    path = tmp_path / "gpu.lock"
    with lease("owner", path):
        path.rename(tmp_path / "old.lock")
        path.touch()
        with pytest.raises(ProjectError, match="ersetzt"), lease("nested", path):
            pytest.fail("replacement accepted")


def test_foreign_process_stays_blocked(tmp_path):
    path = tmp_path / "gpu.lock"
    script = (
        "import fcntl, sys\n"
        "with open(sys.argv[1], 'a+') as handle:\n"
        "    try:\n"
        "        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "    except BlockingIOError:\n"
        "        sys.exit(23)\n"
    )
    with lease("owner", path):
        result = subprocess.run([sys.executable, "-I", "-S", "-c", script, str(path)], timeout=10)
        assert result.returncode == 23
        assert busy(path)
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script, str(path)], timeout=10)
    assert result.returncode == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX flock/fork contract")
def test_fork_cannot_reuse_or_unlock_parent_context(tmp_path):
    path = tmp_path / "gpu.lock"
    guard = lease("parent", path)
    guard.__enter__()
    try:
        pid = os.fork()
        if pid == 0:
            try:
                # Unwinding a copied stack must not truncate/unlock the parent.
                guard.__exit__(None, None, None)
                with lease("child", path):
                    os._exit(1)
            except ProjectError:
                os._exit(0)
            except BaseException:
                os._exit(2)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert busy(path)
        assert json.loads(path.read_text())["job"] == "parent"
    finally:
        guard.__exit__(None, None, None)
    assert not busy(path)


def test_inherited_fd_wrong_file_is_rejected(tmp_path):
    path = tmp_path / "gpu.lock"
    with lease("owner", path), (tmp_path / "other.lock").open("a+") as other:
        with pytest.raises(ProjectError, match="Sperrdatei"), lease("invalid", path, other.fileno()):
            pytest.fail("unrelated descriptor accepted")
