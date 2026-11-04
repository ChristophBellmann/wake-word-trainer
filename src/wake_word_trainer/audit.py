"""Unabhängige Sprachprüfung von Trainingsquellen, ohne Labels zu verändern."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import audio
from .extraction import tokens
from .project import Project, ProjectError
from .recordings import split_for


def inventory(project: Project, extra: list[tuple[str, Path]]) -> list[dict]:
    for _, folder in extra:
        if not folder.is_dir():
            raise ProjectError(f"Zusätzlicher Aufnahmeordner fehlt: {folder}")
    sources = [
        ("positive", project.recordings / "train"),
        ("positive", project.recordings / "eval"),
        ("positive", project.recordings / "hard"),
        ("negative", project.recordings / "negative"),
        *extra,
    ]
    items: dict[str, dict] = {}
    for label, folder in sources:
        for path in audio.wavs(folder):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            item = items.setdefault(digest, {"sha256": digest, "sources": []})
            source = {"label": label, "path": str(path.resolve())}
            if source not in item["sources"]:
                item["sources"].append(source)
    return list(items.values())


def findings(labels: set[str], matches: list[dict], words: list[dict], scores: list[dict]) -> list[str]:
    result = []
    if {"positive", "negative"} <= labels:
        result.append("conflicting_labels")
    if "positive" in labels and not matches:
        result.append("positive_without_stt_match")
    if "negative" in labels and matches:
        result.append("negative_with_stt_match")
    if "unlabelled" in labels:
        result.append("wake_word_candidate" if matches else "negative_candidate" if words else "unclear_audio")
    if "negative" in labels and any(s["peak"] > s["cutoff"] for s in scores):
        result.append("hard_negative")
    if "positive" in labels and any(s["peak"] <= s["cutoff"] for s in scores):
        result.append("missed_positive")
    return result


def transcript_matches(text: str, phrases: list[str]) -> list[dict]:
    """Vollständige Wörter prüfen, unabhängig von fehlerhaften STT-Wortzeiten."""
    words = tokens(text)
    found = []
    for phrase in sorted({tuple(tokens(p)) for p in phrases if tokens(p)}):
        if any(tuple(words[i : i + len(phrase)]) == phrase or words[i] == "".join(phrase) for i in range(len(words))):
            found.append({"phrase": " ".join(phrase), "source": "transcript"})
    return found


class Recognizer:
    def __init__(self, model: str, language: str, threads: int):
        from faster_whisper import WhisperModel

        self.language = None if language == "auto" else language
        self.details = {}
        self.model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)

    def __call__(self, samples: np.ndarray) -> tuple[str, list[dict]]:
        segments, info = self.model.transcribe(
            samples,
            language=self.language,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,
        )
        text, words, statistics = [], [], []
        for segment in segments:
            if segment.no_speech_prob > 0.6 and segment.avg_logprob < -1:
                continue
            statistics.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                    "compression_ratio": segment.compression_ratio,
                }
            )
            text.append(segment.text.strip())
            words.extend(
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for w in segment.words or []
            )
        self.details = {"backend": "faster_whisper", "device": "cpu", "language": info.language, "segments": statistics}
        return " ".join(text), words


class Scorer:
    def __init__(self, path: Path, window: int):
        from microwakeword.inference import Model

        self.model = Model(str(path))
        self.window = window
        manifest = path.with_suffix(".json")
        if not manifest.is_file():
            raise ProjectError(f"Modellmanifest fehlt: {manifest}")
        config = json.loads(manifest.read_text())["micro"]
        self.metadata = {
            "model": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cutoff": float(config["probability_cutoff"]),
            "window": window,
        }

    def __call__(self, samples: np.ndarray) -> dict:
        from microwakeword.audio.audio_utils import generate_features_for_clip

        from .evaluate import moving_average
        from .features import STEP_MS

        padded = np.concatenate([np.zeros(audio.RATE, np.float32), samples, np.zeros(audio.RATE // 2, np.float32)])
        scores = moving_average(
            self.model.predict_spectrogram(generate_features_for_clip(padded, STEP_MS)),
            self.window,
            quantized=getattr(self.model, "is_quantized_model", False),
        )
        return {**self.metadata, "peak": float(scores.max()) if len(scores) else 0.0}


def _validate_runtime(stt_backend: str, stt_device: str, threads: int) -> None:
    if stt_backend not in {"faster_whisper", "torch_whisper"} or stt_device not in {"cpu", "gpu"}:
        raise ProjectError("Unbekannte STT-Engine oder Gerät")
    if stt_backend == "faster_whisper" and stt_device != "cpu":
        raise ProjectError("GPU-STT verwendet torch_whisper; faster_whisper bleibt CPU")
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ProjectError("Datenprüfung benötigt 1 bis 32 Threads")


def run(
    project: Project,
    extra: list[tuple[str, Path]],
    output: Path,
    stt_model: str = "small",
    models: list[Path] | None = None,
    phrases: list[str] | None = None,
    threads: int = 4,
    teachers: list[Path] | None = None,
    stt_backend: str = "faster_whisper",
    stt_device: str = "cpu",
    stt_language: str | None = None,
    stt_cache: Path | None = None,
    verification_model: str = "",
) -> dict:
    from contextlib import nullcontext

    from .gpu import lease

    _validate_runtime(stt_backend, stt_device, threads)
    with lease("STT-Datenprüfung") if stt_device == "gpu" else nullcontext():
        return _run(
            project,
            extra,
            output,
            stt_model,
            models,
            phrases,
            threads,
            teachers,
            stt_backend,
            stt_device,
            stt_language,
            stt_cache,
            verification_model,
        )


def _run(
    project: Project,
    extra: list[tuple[str, Path]],
    output: Path,
    stt_model: str = "small",
    models: list[Path] | None = None,
    phrases: list[str] | None = None,
    threads: int = 4,
    teachers: list[Path] | None = None,
    stt_backend: str = "faster_whisper",
    stt_device: str = "cpu",
    stt_language: str | None = None,
    stt_cache: Path | None = None,
    verification_model: str = "",
) -> dict:
    """Schreibt private JSONL-Berichte; wiederholt nur geänderte/fehlgeschlagene Clips."""
    _validate_runtime(stt_backend, stt_device, threads)
    language = stt_language or project.config["language"]
    items = inventory(project, extra)
    if not items:
        raise ProjectError("Keine Aufnahmen für die Datenprüfung vorhanden")
    output.mkdir(parents=True, exist_ok=True)
    phrases = list(dict.fromkeys([project.config["wake_word"], *project.config["tts"]["phrases"], *(phrases or [])]))
    scorers = [Scorer(path, int(project.config["evaluation"]["sliding_window_size"])) for path in models or []]
    if teachers:
        from .teacher import Teacher

        teacher_scorers = [Teacher(path, device=stt_device, threads=threads) for path in teachers]
    else:
        teacher_scorers = []
    config = {
        "stt_model": stt_model,
        "language": language,
        "stt_backend": stt_backend,
        "stt_device": stt_device,
        "verification_model": verification_model,
        "phrases": phrases,
        "models": [s.metadata for s in scorers],
        "version": 2,
    }
    if teacher_scorers:
        config["teachers"] = [s.metadata for s in teacher_scorers]
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    cache = {}
    report = output / "clips.jsonl"
    if report.is_file():
        for line in report.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # nach einem unterbrochenen Lauf
            if row.get("signature") == signature and not row.get("error"):
                cache[row["sha256"]] = row
    recognizer = None
    verifier = None
    counts: Counter = Counter()
    rows = []
    # Einzelne Ergebnisse bleiben bei einem Abbruch für den nächsten Aufruf erhalten.
    report.touch(mode=0o600, exist_ok=True)
    report.chmod(0o600)
    with report.open("a", encoding="utf-8") as stream:
        for index, item in enumerate(items, 1):
            row = {**cache.get(item["sha256"], {}), **item, "signature": signature}
            if item["sha256"] not in cache:
                try:
                    samples = audio.read(item["sources"][0]["path"])
                    if len(samples) == 0 or not np.isfinite(samples).all():
                        raise ValueError("Leere oder ungültige Audiodaten")
                    if recognizer is None:
                        if stt_backend == "torch_whisper":
                            from .stt import TorchRecognizer

                            recognizer = TorchRecognizer(
                                stt_model, language, threads, stt_device, stt_cache or project.path("models", "stt")
                            )
                        else:
                            recognizer = Recognizer(stt_model, language, threads)
                    transcript, words = recognizer(samples)
                    if verification_model:
                        if verifier is None:
                            verifier = Recognizer(verification_model, language, threads)
                        verified_text, verified_words = verifier(samples)
                        row["verification"] = {
                            "transcript": verified_text,
                            "words": verified_words,
                            "stt": getattr(verifier, "details", {}),
                        }
                    row.update(
                        stt=getattr(recognizer, "details", {}),
                        duration_seconds=len(samples) / audio.RATE,
                        transcript=transcript,
                        words=words,
                        matches=transcript_matches(transcript, phrases),
                        scores=[s(samples) for s in scorers],
                        teachers=[s(samples) for s in teacher_scorers],
                    )
                except Exception as err:
                    row["error"] = f"{type(err).__name__}: {err}"
            labels = {s["label"] for s in item["sources"]}
            row["findings"] = (
                ["processing_error"]
                if row.get("error")
                else findings(
                    labels,
                    row["matches"],
                    row["words"],
                    row["scores"],
                )
            )
            row["prospective_split"] = split_for(item["sha256"], float(project.config["recordings"]["eval_share"]))
            row["requires_listening"] = bool(row["findings"])
            counts.update(row["findings"])
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 25 == 0 or index == len(items):
                print(f"Datenprüfung: {index}/{len(items)} Aufnahmen", flush=True)
    # Ein eindeutiger aktueller Bericht nach erfolgreichem Abschluss.
    replacement = report.with_suffix(".jsonl.tmp")
    replacement.touch(mode=0o600, exist_ok=True)
    replacement.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    replacement.replace(report)
    summary = {"config": config, "unique_clips": len(rows), "findings": dict(counts), "labels_changed": 0}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
