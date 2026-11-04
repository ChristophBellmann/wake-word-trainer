"""Lokale mehrsprachige Whisper-Prüfung auf CPU, CUDA oder ROCm-PyTorch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .project import ProjectError


class TorchRecognizer:
    def __init__(self, model: str, language: str, threads: int, device: str, cache: Path):
        import torch
        import whisper

        if device not in {"cpu", "gpu"} or not 1 <= threads <= 32:
            raise ProjectError("STT benötigt device cpu/gpu und 1 bis 32 Threads")
        torch.set_num_threads(threads)
        if device == "gpu" and not torch.cuda.is_available():
            raise ProjectError("GPU-STT angefordert, aber PyTorch erkennt keine CUDA/ROCm-GPU")
        selected = "cuda" if device == "gpu" else "cpu"
        probe = torch.ones((16, 16), device=selected) @ torch.ones((16, 16), device=selected)
        if probe.device.type != selected or float(probe.sum().cpu()) != 4096.0:
            raise ProjectError("STT-Rechenprobe auf angefordertem Gerät fehlgeschlagen")
        cache.mkdir(parents=True, exist_ok=True)
        self.model = whisper.load_model(model, device=selected, download_root=str(cache))
        self.language = None if language == "auto" else language
        self.device = selected
        self.details = {}
        self.metadata = {
            "backend": "torch_whisper",
            "model": model,
            "device": selected,
            "torch_version": torch.__version__,
            "hip": torch.version.hip,
        }

    def __call__(self, samples: np.ndarray) -> tuple[str, list[dict]]:
        result = self.model.transcribe(
            samples,
            language=self.language,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            fp16=self.device == "cuda",
            verbose=None,
        )
        words = [dict(word) for segment in result.get("segments", []) for word in segment.get("words", [])]
        segments = [
            {key: segment.get(key) for key in ("start", "end", "avg_logprob", "no_speech_prob", "compression_ratio")}
            for segment in result.get("segments", [])
        ]
        self.details = {**self.metadata, "language": result.get("language"), "segments": segments}
        return result.get("text", "").strip(), words
