"""ROCm auto-loader must not alter a CUDA/CPU framework environment."""

from types import SimpleNamespace

import pytest

from wake_word_trainer.gpu import _tensorflow_rocm_build, environment


@pytest.mark.parametrize("cuda,rocm", [(True, False), (False, True), (False, False)])
def test_static_framework_metadata_without_import(tmp_path, monkeypatch, cuda, rocm):
    package = tmp_path / "tensorflow"
    metadata = package / "python/platform/build_info.py"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        f"import collections\nbuild_info = collections.OrderedDict([('is_cuda_build', {cuda}), ('is_rocm_build', {rocm})])\n"
    )
    monkeypatch.setattr(
        "wake_word_trainer.gpu.importlib.util.find_spec",
        lambda name: SimpleNamespace(origin=str(package / "__init__.py")),
    )
    assert _tensorflow_rocm_build() is rocm


def test_cuda_does_not_load_installed_system_rocm(monkeypatch):
    monkeypatch.setattr("wake_word_trainer.gpu._tensorflow_rocm_build", lambda: False)
    monkeypatch.setattr("wake_word_trainer.gpu._python_rocm_sdk", lambda: False)
    original = {"PATH": "/cuda/venv/bin", "LD_LIBRARY_PATH": "/cuda/lib"}
    result = environment(original)
    assert result["PATH"] == original["PATH"]
    assert result["LD_LIBRARY_PATH"] == original["LD_LIBRARY_PATH"]
    assert "ROCM_PATH" not in result and "LD_PRELOAD" not in result
    assert not any(key.startswith("TF_ROCM_") for key in result)
