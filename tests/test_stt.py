"""Device probe, language selection and local transcription metadata."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wake_word_trainer.project import ProjectError
from wake_word_trainer.stt import TorchRecognizer


def fake_torch(monkeypatch, *, available=True, actual_device="cuda"):
    class Probe:
        device = SimpleNamespace(type=actual_device)

        def __matmul__(self, other):
            return self

        def sum(self):
            return self

        def cpu(self):
            return 4096.0

    module = SimpleNamespace(
        set_num_threads=lambda n: None,
        cuda=SimpleNamespace(is_available=lambda: available),
        ones=lambda *a, **kw: Probe(),
        __version__="test",
        version=SimpleNamespace(hip="test-rocm"),
    )
    monkeypatch.setitem(sys.modules, "torch", module)


def test_rocm_uses_pytorch_cuda_namespace_and_auto_language(tmp_path, monkeypatch):
    fake_torch(monkeypatch)
    requests = []

    class Model:
        def transcribe(self, samples, **kwargs):
            requests.append(kwargs)
            return {
                "text": " start vacuuming ",
                "language": "en",
                "segments": [
                    {
                        "start": 0,
                        "end": 1,
                        "avg_logprob": -0.1,
                        "no_speech_prob": 0.01,
                        "words": [{"word": "start", "probability": 0.9}],
                    }
                ],
            }

    loads = []
    monkeypatch.setitem(
        sys.modules, "whisper", SimpleNamespace(load_model=lambda *a, **kw: (loads.append(kw), Model())[1])
    )
    recognizer = TorchRecognizer("large-v3", "auto", 4, "gpu", tmp_path / "cache")
    text, words = recognizer(np.zeros(16000, np.float32))
    assert text == "start vacuuming" and len(words) == 1
    assert loads[0]["device"] == "cuda"
    assert requests[0]["language"] is None and requests[0]["condition_on_previous_text"] is False
    assert recognizer.details["language"] == "en" and recognizer.details["hip"] == "test-rocm"


@pytest.mark.parametrize("available,actual_device", [(False, "cuda"), (True, "cpu")])
def test_gpu_request_refuses_absent_gpu_or_cpu_fallback(tmp_path, monkeypatch, available, actual_device):
    fake_torch(monkeypatch, available=available, actual_device=actual_device)
    monkeypatch.setitem(
        sys.modules, "whisper", SimpleNamespace(load_model=lambda *a, **kw: pytest.fail("model loaded"))
    )
    with pytest.raises(ProjectError):
        TorchRecognizer("large-v3", "auto", 4, "gpu", tmp_path / "cache")
