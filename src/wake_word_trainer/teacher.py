"""Optionaler PyTorch-Teacher mit eigenem Log-Mel-Frontend, unabhängig vom Lab."""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import audio
from .gpu import lease, torch_device
from .project import Project, ProjectError

FORMAT = "wake-word-trainer-logmel-v1"


@dataclass(frozen=True)
class Parameters:
    sample_rate: int = 16000
    segment_seconds: float = 1.5
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 40
    fmin: float = 20.0
    fmax: float = 7600.0


def network(dropout: float = 0.2):
    import torch.nn as nn

    # Zustandsnamen und Architektur des reparierten Log-Mel-v2-Teachers.
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 24, 3, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(24, 48, 3, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(48, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, samples):
            return self.net(samples)

    return Net()


class Frontend:
    """Trainingstreue STFT-Log-Mel-Features; niemals microfrontend-MMAPs einlesen."""

    def __init__(self, params: Parameters):
        self.params = params
        if params.sample_rate != audio.RATE or params.win_length > params.n_fft:
            raise ProjectError("Teacher benötigt 16 kHz und win_length <= n_fft")
        self.length = round(params.segment_seconds * params.sample_rate)
        if self.length < params.win_length or not 0 <= params.fmin < params.fmax <= params.sample_rate / 2:
            raise ProjectError("Ungültige Log-Mel-Parameter")
        self.window = np.hanning(params.win_length).astype(np.float32)
        mels = np.linspace(
            2595 * np.log10(1 + params.fmin / 700), 2595 * np.log10(1 + params.fmax / 700), params.n_mels + 2
        )
        hz = 700 * (10 ** (mels / 2595) - 1)
        bins = np.floor((params.n_fft + 1) * hz / params.sample_rate).astype(int)
        self.filters = np.zeros((params.n_mels, params.n_fft // 2 + 1), np.float32)
        for i in range(1, params.n_mels + 1):
            left, center, right = bins[i - 1 : i + 2]
            center, right = max(center, left + 1), max(right, center + 1)
            for j in range(left, center):
                self.filters[i - 1, j] = (j - left) / float(center - left)
            for j in range(center, right):
                self.filters[i - 1, j] = (right - j) / float(right - center)

    def clip(self, samples: np.ndarray) -> np.ndarray:
        p = self.params
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(samples, p.win_length)[:: p.hop_length]
        spec = np.fft.rfft(frames * self.window[None, :], n=p.n_fft, axis=-1)
        power = (spec.real**2 + spec.imag**2).astype(np.float32)
        return np.log(np.maximum(power @ self.filters.T, 1e-10)).T.astype(np.float32)

    def windows(self, samples: np.ndarray, hop_seconds: float = 0.1) -> np.ndarray:
        if not len(samples) or not np.isfinite(samples).all():
            raise ProjectError("Leere oder ungültige Teacher-Audiodaten")
        samples = np.pad(samples, (0, max(0, self.length - len(samples))))
        last = len(samples) - self.length
        starts = list(range(0, last + 1, max(1, round(hop_seconds * self.params.sample_rate))))
        if starts[-1] != last:
            starts.append(last)
        return np.stack([self.clip(samples[s : s + self.length]) for s in starts])


def save(path: Path, model, params: Parameters, **metadata) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "feature_space": "logmel",
            "features": asdict(params),
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            **metadata,
        },
        path,
    )
    path.chmod(0o600)


def import_checkpoint(source: Path, config: Path, target: Path) -> None:
    import torch
    import yaml

    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    if settings.get("data", {}).get("negative_mmap_roots"):
        raise ProjectError("Legacy-Teacher mit gemischten Feature-Räumen wird nicht als Log-Mel-Teacher importiert")
    features = dict(settings["features"])
    features["fmin"] = features.pop("fmin_hz", 20.0)
    features["fmax"] = features.pop("fmax_hz", 7600.0)
    params = Parameters(**features)
    Frontend(params)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    model = network()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    save(target, model, params, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())


class Teacher:
    def __init__(self, path: Path, threads: int = 2, device: str = "auto"):
        import torch

        torch.set_num_threads(threads)
        self.device = torch_device(device)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != FORMAT or checkpoint.get("feature_space") != "logmel":
            raise ProjectError("Kein portabler Log-Mel-Teacher; zuerst teacher import mit der Original-Konfiguration")
        self.frontend = Frontend(Parameters(**checkpoint["features"]))
        self.model = network()
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        guard = lease("teacher-initialize") if self.device.type == "cuda" else contextlib.nullcontext()
        with guard:
            self.model.to(self.device)
        self.model.eval()
        self.metadata = {
            "model": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "feature_space": "logmel",
            "device": str(self.device),
        }

    def __call__(self, samples: np.ndarray) -> dict:
        import torch

        features = self.frontend.windows(samples)
        scores = []
        guard = lease("teacher-scoring") if self.device.type == "cuda" else contextlib.nullcontext()
        with guard, torch.inference_mode():
            for start in range(0, len(features), 128):
                batch = torch.from_numpy(features[start : start + 128]).unsqueeze(1).to(self.device)
                scores.extend(torch.sigmoid(self.model(batch)).cpu().numpy().reshape(-1).tolist())
        return {
            **self.metadata,
            "peak": float(np.max(scores)),
            "p95": float(np.quantile(scores, 0.95)),
            "windows": len(scores),
        }


def train(project: Project, epochs: int = 30, batch_size: int = 128, seed: int = 1337, device: str = "auto") -> Path:
    selected = torch_device(device)
    guard = lease("teacher-training") if selected.type == "cuda" else contextlib.nullcontext()
    with guard:
        return _train(project, epochs, batch_size, seed, selected)


def _train(project: Project, epochs: int, batch_size: int, seed: int, device) -> Path:
    """Geprüfte Labels; Holdouts bleiben aus dem Optimierer. GPU ausdrücklich wählbar."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    frontend = Frontend(Parameters())
    model = network(0.25).to(device)
    print(f"Teacher training device: {device}", flush=True)

    def dataset(split):
        positives = audio.wavs(project.recordings / split)
        negatives = audio.wavs(project.recordings / "negative" / split)
        if not positives or not negatives:
            raise ProjectError(f"Teacher benötigt positive und negative Aufnahmen in {split}")
        # Für diese erste portable Variante: ein vollständiges Fenster pro Clip.
        # Die Bewertung prüft dagegen alle gleitenden Fenster.
        features, labels = [], []
        for truth, paths in ((1.0, positives), (0.0, negatives)):
            for path in paths:
                samples = audio.read(path)[-frontend.length :]
                samples = np.pad(samples, (0, max(0, frontend.length - len(samples))))
                features.append(frontend.clip(samples))
                labels.append(truth)
        return TensorDataset(
            torch.from_numpy(np.stack(features)).unsqueeze(1), torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
        ), len(negatives) / len(positives)

    training, ratio = dataset("train")
    validation, _ = dataset("eval")
    train_hashes = {
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in audio.wavs(project.recordings / "train", project.recordings / "negative" / "train")
    }
    eval_hashes = {
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in audio.wavs(project.recordings / "eval", project.recordings / "negative" / "eval")
    }
    if train_hashes & eval_hashes:
        raise ProjectError("Teacher-Training verweigert: identischer Inhalt in Training und Bewertung")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio], device=device))
    target = project.path("teacher", "model.pt")
    best, stale, history = float("inf"), 0, []
    for epoch in range(epochs):
        model.train()
        for features, labels in DataLoader(
            training, batch_size=batch_size, shuffle=True, pin_memory=device.type == "cuda"
        ):
            features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            criterion(model(features), labels).backward()
            optimizer.step()
        model.eval()
        total = 0.0
        with torch.inference_mode():
            for features, labels in DataLoader(validation, batch_size=batch_size, pin_memory=device.type == "cuda"):
                features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                total += float(criterion(model(features), labels)) * len(labels)
        loss = total / len(validation)
        history.append({"epoch": epoch + 1, "validation_loss": loss})
        print(f"Teacher: Epoche {epoch + 1}/{epochs}, Prüfloss {loss:.4f}", flush=True)
        if loss < best:
            best, stale = loss, 0
            save(target, model, frontend.params, seed=seed, history=history, training_device=str(device))
        else:
            stale += 1
        if stale >= 6:
            break
    project.path("teacher", "training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return target
