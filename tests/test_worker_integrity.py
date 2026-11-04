"""Synthetic source/worker fixtures; never import or initialize NN frameworks."""

import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from wake_word_trainer import cli
from wake_word_trainer.project import Project
from wake_word_trainer.service import Service, _source_manifest, _worker_command


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source" / "wake_word_trainer"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("VALUE = 'current'\n")
    (root / "__main__.py").write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "import wake_word_trainer as package\n"
        "print(json.dumps({'source': str(Path(package.__file__).resolve().parent), "
        "'value': package.VALUE, 'args': sys.argv[1:]}))\n"
    )
    return root


def run_worker(source, manifest=None, env=None, cwd=None):
    command = _worker_command(
        Path(sys.executable), source, manifest if manifest is not None else _source_manifest(source)
    )
    return subprocess.run([*command, "fixture-argument"], env=env, cwd=cwd, text=True, capture_output=True, timeout=15)


def test_legitimate_worker_uses_controller_source_with_poisoned_loader(source, tmp_path):
    foreign = tmp_path / "foreign"
    package = foreign / "wake_word_trainer"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('wrong trainer source')\n")
    (foreign / "sitecustomize.py").write_text("raise RuntimeError('wrong sitecustomize')\n")
    env = dict(os.environ, PYTHONPATH=str(foreign), PYTHONHOME=str(foreign), PYTHONUSERBASE=str(foreign))
    result = run_worker(source, env=env, cwd=foreign)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"source": str(source), "value": "current", "args": ["fixture-argument"]}


@pytest.mark.parametrize("change", ["modified", "missing", "added", "empty_binding", "wrong_source"])
def test_source_mismatch_fails_before_package_execution(source, tmp_path, change):
    manifest = _source_manifest(source)
    if change == "modified":
        (source / "__init__.py").write_text("raise AssertionError('must not execute')\n")
    elif change == "missing":
        (source / "__init__.py").unlink()
    elif change == "added":
        (source / "unexpected.py").write_text("raise AssertionError('must not execute')\n")
    elif change == "empty_binding":
        manifest = {}
    else:
        source = tmp_path / "other_source"
        source.mkdir()
        (source / "__init__.py").write_text("raise AssertionError('must not execute')\n")
    result = run_worker(source, manifest)
    assert result.returncode != 0
    assert "Worker source binding missing or changed" in result.stderr
    assert result.stdout == ""
    assert "AssertionError" not in result.stderr


def test_stale_timestamp_valid_bytecode_cannot_replace_verified_source(source):
    path = source / "__init__.py"
    source_bytes = path.read_bytes()
    stamp = path.stat()
    # Both strings must have identical length for a timestamp/size-valid cache.
    path.write_bytes(source_bytes.replace(b"current", b"outdate"))
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    py_compile.compile(str(path), doraise=True)
    path.write_bytes(source_bytes)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    result = run_worker(source)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["value"] == "current"


def test_late_import_rechecks_bytes_before_execution(source):
    (source / "late.py").write_text("VALUE = 'before'\n")
    (source / "__init__.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('late.py').write_text(\"raise AssertionError('must not execute')\\n\")\n"
        "from . import late\n"
    )
    result = run_worker(source)
    assert result.returncode != 0
    assert "Worker source binding missing or changed" in result.stderr
    assert result.stdout == ""
    assert "AssertionError" not in result.stderr


def test_missing_worker_binding_is_rejected(source):
    command = _worker_command(Path(sys.executable), source, _source_manifest(source))
    result = subprocess.run(command[:8], text=True, capture_output=True, timeout=15)
    assert result.returncode != 0
    assert "Worker source binding missing or changed" in result.stderr


def configured_service(tmp_path, monkeypatch, **profile):
    project = Project.create(tmp_path / "project", "Hey Fixture")
    token = tmp_path / "service.token"
    token.write_text("synthetic-token-for-test-only")
    service = Service(
        {"project": str(project.root), "token_file": str(token), "profiles": {"review": {"kind": "review", **profile}}}
    )
    monkeypatch.setattr(service.resources, "release", lambda: None)
    monkeypatch.setattr(service.resources, "acquire", lambda: None)
    monkeypatch.setattr(
        "wake_word_trainer.service.threading.Thread", lambda **kwargs: SimpleNamespace(start=lambda: None)
    )
    return service


def test_relative_worker_interpreter_is_rejected_before_resources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("relative-python").touch()
    service = configured_service(tmp_path, monkeypatch, worker_python="relative-python")
    monkeypatch.setattr(service.system, "before_start", lambda: pytest.fail("resources checked for invalid worker"))
    code, response = service.start("review")
    assert code == 400 and "absolute" in response["error"]
    assert not service.project.path("state.json").exists()


@pytest.mark.parametrize("isolated", [False, True])
def test_service_cleans_python_loader_and_preserves_gpu_contract(tmp_path, monkeypatch, isolated):
    service = configured_service(tmp_path, monkeypatch, worker_python=sys.executable, isolated_runtime=isolated)
    original = {
        "PYTHONPATH": "/synthetic/poison",
        "PYTHONHOME": "/synthetic/poison",
        "PYTHONINSPECT": "1",
        "LD_LIBRARY_PATH": "/synthetic/gpu",
        "LD_PRELOAD": "/synthetic/preload",
        "ROCM_PATH": "/synthetic/rocm",
        "WAKE_WORD_TRAINER_RUNTIME_READY": "1",
    }
    monkeypatch.setattr("wake_word_trainer.service.os.environ", original)
    monkeypatch.setattr("wake_word_trainer.gpu.environment", lambda: original.copy())
    captured = {}

    def launch(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr("wake_word_trainer.service.subprocess.Popen", launch)
    assert service.start("review")[0] == 202
    command, env = captured["command"], captured["env"]
    assert command[:6] == [sys.executable, "-I", "-S", "-B", "-u", "-c"]
    assert command[7] == str(service.worker_source)
    assert json.loads(command[8]) == service.worker_manifest
    assert not any(key.startswith("PYTHON") for key in env)
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "ROCM_PATH", "WAKE_WORD_TRAINER_RUNTIME_READY"):
        assert (key not in env) if isolated else env[key] == original[key]
    assert all(env[key] == "-1" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))


def test_controller_rejects_changed_source_before_resources(tmp_path, monkeypatch):
    service = configured_service(tmp_path, monkeypatch)
    monkeypatch.setattr("wake_word_trainer.service._source_manifest", lambda root: {})
    monkeypatch.setattr(service.system, "before_start", lambda: pytest.fail("resources checked for changed source"))
    assert service.start("review")[0] == 503
    assert not service.project.path("state.json").exists()


def test_runtime_reexec_preserves_original_bootstrap_and_binding(monkeypatch):
    original = [sys.executable, "-I", "-S", "-B", "-u", "-c", "fixture-bootstrap", "/source", "{}", "review"]
    monkeypatch.setattr(sys, "orig_argv", original)
    monkeypatch.setattr(sys, "argv", ["wake_word_trainer", "review"])
    monkeypatch.delenv("WAKE_WORD_TRAINER_RUNTIME_READY", raising=False)
    monkeypatch.setattr(
        cli, "parser", lambda: SimpleNamespace(parse_args=lambda argv: SimpleNamespace(command="review"))
    )
    monkeypatch.setattr("wake_word_trainer.gpu.environment", lambda: {"LD_LIBRARY_PATH": "/synthetic/runtime"})
    captured = {}

    def reexec(executable, command, env):
        captured.update(executable=executable, command=command, env=env)
        raise SystemExit(99)

    monkeypatch.setattr(cli.os, "execve", reexec)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 99
    assert captured["command"] == original
    assert captured["env"]["WAKE_WORD_TRAINER_RUNTIME_READY"] == "1"


def test_api_example_reads_existing_token_file_in_process(monkeypatch):
    import io
    import urllib.request

    document = (Path(__file__).resolve().parents[1] / "docs/api.md").read_text()
    snippet = document.split("python3 - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    token = "synthetic-token-for-documentation-test"
    captured = {}

    def read_token(path, *args, **kwargs):
        assert str(path).endswith("/.config/wake-word-trainer/service.token")
        return token + "\n"

    def opener(handler):
        handler = handler() if isinstance(handler, type) else handler
        assert handler.redirect_request(None, None, 302, None, None, "http://example.invalid") is None

        def request(req, timeout):
            captured["authorization"] = req.get_header("Authorization")
            assert req.full_url == "http://localhost:10701/v1/status" and timeout == 10
            return io.BytesIO(b'{"state": "fixture"}')

        return SimpleNamespace(open=request)

    monkeypatch.setattr(Path, "read_text", read_token)
    monkeypatch.setattr(urllib.request, "build_opener", opener)
    exec(compile(snippet, "docs/api.md", "exec"), {})
    assert captured["authorization"] == f"Bearer {token}"
    assert all(token not in argument for argument in sys.argv)
