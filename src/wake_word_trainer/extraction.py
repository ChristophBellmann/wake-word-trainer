"""Find configured wake-word phrases using local, independent speech recognition."""

from __future__ import annotations

import io
import math
import re
import unicodedata
import wave
from itertools import pairwise

MAX_BYTES = 4 * 1024 * 1024


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[^\W_]+", text, re.UNICODE)


def validate_audio(body: bytes) -> float:
    if not 44 <= len(body) <= MAX_BYTES:
        raise ValueError("invalid audio size")
    try:
        with wave.open(io.BytesIO(body), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (1, 2, "NONE"):
                raise ValueError("expected mono PCM16 WAV")
            if wav.getframerate() not in (16000, 48000):
                raise ValueError("expected 16 or 48 kHz")
            duration = wav.getnframes() / wav.getframerate()
            if not 0.5 <= duration <= 120 or len(wav.readframes(wav.getnframes())) != wav.getnframes() * 2:
                raise ValueError("invalid audio duration")
            return duration
    except (wave.Error, EOFError) as err:
        raise ValueError("invalid WAV") from err


def find_phrases(words: list[dict], phrases: list[str], duration: float, config: dict) -> list[dict]:
    """Match complete consecutive words; never infer a phrase from silence alone."""
    accepted = sorted({tuple(tokens(p)) for p in phrases if tokens(p)}, key=len, reverse=True)
    padding = float(config.get("padding_ms", 200)) / 1000
    threshold = float(config.get("min_probability", 0.35))
    timed = []
    for word in words:
        start, end, probability = (float(word[k]) for k in ("start", "end", "probability"))
        if not all(math.isfinite(v) for v in (start, end, probability)) or not 0 <= start < end <= duration + 0.05:
            continue
        for token in tokens(word["word"]):
            timed.append({"token": token, "start": start, "end": min(end, duration), "probability": probability})
    matches, i = [], 0
    while i < len(timed):
        for phrase in accepted:
            length = 1 if timed[i]["token"] == "".join(phrase) else len(phrase)
            part = timed[i : i + length]
            if length != 1 and tuple(w["token"] for w in part) != phrase:
                continue
            if length == 1 and timed[i]["token"] != "".join(phrase):
                continue
            if any(b["start"] < a["start"] or b["start"] - a["end"] > 1 for a, b in pairwise(part)):
                continue
            confidence = sum(w["probability"] for w in part) / len(part)
            if confidence < threshold or not 0.2 <= part[-1]["end"] - part[0]["start"] <= 3:
                continue
            start, end = max(0, part[0]["start"] - padding), min(duration, part[-1]["end"] + padding)
            # Keep padding out of neighbouring speech, including other repetitions.
            if i:
                start = max(start, (timed[i - 1]["end"] + part[0]["start"]) / 2)
            if i + length < len(timed):
                end = min(end, (part[-1]["end"] + timed[i + length]["start"]) / 2)
            if end - start < 0.5:
                continue
            matches.append(
                {
                    "start_ms": round(start * 1000),
                    "end_ms": round(end * 1000),
                    "phrase": " ".join(phrase),
                    "confidence": round(confidence, 4),
                }
            )
            i += length
            break
        else:
            i += 1
    return matches


class Extractor:
    def __init__(self, config: dict, language: str):
        self.config = config
        self.language = language
        self.model = None

    def _speech_regions(self, samples):
        from faster_whisper.vad import get_speech_timestamps

        return get_speech_timestamps(
            samples,
            min_silence_duration_ms=int(self.config.get("silence_ms", 500)),
            min_speech_duration_ms=150,
            max_speech_duration_s=10,
            speech_pad_ms=500,
        )

    def extract(self, body: bytes, phrases: list[str]) -> dict:
        import numpy as np
        from scipy.signal import resample_poly

        duration = validate_audio(body)
        with wave.open(io.BytesIO(body), "rb") as wav:
            rate = wav.getframerate()
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
        if rate == 48000:
            samples = resample_poly(samples, 1, 3)
        regions = self._speech_regions(samples)
        if not regions:
            return {"segments": [], "duration_ms": round(duration * 1000)}
        if self.model is None:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                str(self.config.get("model", "small")),
                device="cpu",
                compute_type="int8",
                cpu_threads=int(self.config.get("cpu_threads", 4)),
            )
        words = []
        # Decode speech regions separately. Concatenating identical repetitions
        # lets Whisper collapse them into one phrase and lose training examples.
        for region in regions:
            offset = region["start"] / 16000
            segments, _ = self.model.transcribe(
                samples[region["start"] : region["end"]],
                language=self.language,
                beam_size=5,
                word_timestamps=True,
                without_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            for segment in segments:
                if segment.no_speech_prob > 0.6 and segment.avg_logprob < -1:
                    continue
                words.extend(
                    {"word": w.word, "start": w.start + offset, "end": w.end + offset, "probability": w.probability}
                    for w in segment.words or []
                )
        return {"segments": find_phrases(words, phrases, duration, self.config), "duration_ms": round(duration * 1000)}
