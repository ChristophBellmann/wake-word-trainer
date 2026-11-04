"""Begrenzte Negativfreigabe: bekannte Phrase, zwei STT und geprüfter Teacher."""

from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

from .audit import transcript_matches
from .extraction import tokens
from .project import ProjectError


def load_policy(path: Path) -> dict:
    policy = yaml.safe_load(path.read_text())
    if not isinstance(policy, dict):
        raise ProjectError("Negativrichtlinie muss eine Zuordnung sein")
    allowed = {
        "known_negative_phrases",
        "allowed_teacher_hashes",
        "teacher_calibrations",
        "min_word_probability",
        "max_teacher_score",
    }
    if set(policy) - allowed:
        raise ProjectError("Unbekannte Einstellung in Negativrichtlinie")
    phrases = policy.get("known_negative_phrases", [])
    hashes = policy.get("allowed_teacher_hashes", [])
    if (
        not isinstance(phrases, list)
        or not 1 <= len(phrases) <= 100
        or any(not isinstance(p, str) or not tokens(p) or len(p) > 200 for p in phrases)
    ):
        raise ProjectError("Negativrichtlinie benötigt bekannte negative Phrasen")
    if (
        not isinstance(hashes, list)
        or not hashes
        or any(not isinstance(h, str) or not re.fullmatch("[a-f0-9]{64}", h) for h in hashes)
    ):
        raise ProjectError("Negativrichtlinie benötigt geprüfte Teacher-Hashes")
    calibrations = policy.get("teacher_calibrations", {})
    if not isinstance(calibrations, dict):
        raise ProjectError("Teacher-Kalibrierungen müssen eine Zuordnung sein")
    for digest, calibration in calibrations.items():
        if (
            digest not in hashes
            or not isinstance(calibration, dict)
            or set(calibration) != {"target_phrases", "evidence"}
            or not _phrases(calibration.get("target_phrases"))
            or not isinstance(calibration.get("evidence"), str)
            or not calibration["evidence"].strip()
        ):
            raise ProjectError("Teacher-Kalibrierung benötigt freigegebenen Hash, Ziele und Prüfbeleg")
    result = {"min_word_probability": 0.85, "max_teacher_score": 0.1, "teacher_calibrations": {}, **policy}
    for key in ("min_word_probability", "max_teacher_score"):
        value = result[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ProjectError("Ungültige Konfidenzgrenze in Negativrichtlinie")
    return result


def _phrases(value) -> set[tuple[str, ...]]:
    if not isinstance(value, list) or not value:
        return set()
    if any(not isinstance(p, str) or not tokens(p) or len(p) > 200 for p in value):
        return set()
    return {tuple(tokens(p)) for p in value}


def _number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _confident(result: dict, minimum: float) -> bool:
    words, stt = result.get("words"), result.get("stt")
    if not isinstance(words, list) or not words or not isinstance(stt, dict) or stt.get("error"):
        return False
    segments = stt.get("segments")
    if not isinstance(segments, list) or not segments:
        return False
    recognized = []
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get("word"), str):
            return False
        recognized.extend(tokens(word["word"]))
        probability = word.get("probability")
        if not _number(probability) or not minimum <= probability <= 1:
            return False
    if recognized != tokens(result["transcript"]):
        return False  # probabilities must cover the entire recognized text
    for segment in segments:
        if not isinstance(segment, dict):
            return False
        logprob, silence = segment.get("avg_logprob"), segment.get("no_speech_prob")
        if not _number(logprob) or not _number(silence) or not -0.5 <= logprob <= 0 or not 0 <= silence <= 0.4:
            return False
    return True


def decide(row: dict, positive_phrases: list[str], policy: dict) -> dict:
    result = {"eligible": False, "reason": "requires_review"}
    if row.get("origin_ambiguous"):
        return {"eligible": False, "reason": "ambiguous_origin"}
    if row.get("error") or row.get("category") != "triggers":
        return result
    stt, verification = row.get("stt"), row.get("verification")
    if not isinstance(stt, dict) or stt.get("backend") != "torch_whisper":
        return result
    if not isinstance(verification, dict) or verification.get("error"):
        return result
    verified_stt = verification.get("stt")
    if not isinstance(verified_stt, dict) or verified_stt.get("backend") != "faster_whisper":
        return result
    texts = [obs.get("transcript") for obs in (row, verification)]
    if any(not isinstance(text, str) or not tokens(text) for text in texts):
        return result
    if tokens(texts[0]) != tokens(texts[1]):
        return {"eligible": False, "reason": "wake_word_or_disagreement"}
    teachers = row.get("teachers")
    if not isinstance(teachers, list) or not teachers:
        return result
    project_targets = _phrases(positive_phrases)
    if not project_targets:
        return {"eligible": False, "reason": "teacher_target_or_calibration_unclear"}
    protected_phrases = list(positive_phrases)
    for teacher in teachers:
        if not isinstance(teacher, dict) or teacher.get("error"):
            return result
        digest = teacher.get("sha256")
        if not isinstance(digest, str) or digest not in policy["allowed_teacher_hashes"]:
            return result
        targets = _phrases(teacher.get("target_phrases"))
        calibration = policy.get("teacher_calibrations", {}).get(digest, {})
        if (
            not targets
            or not project_targets <= targets
            or not isinstance(calibration, dict)
            or _phrases(calibration.get("target_phrases")) != targets
            or not isinstance(calibration.get("evidence"), str)
            or not calibration["evidence"].strip()
        ):
            return {"eligible": False, "reason": "teacher_target_or_calibration_unclear"}
        protected_phrases.extend(teacher["target_phrases"])
        value = teacher.get("peak")
        if not _number(value) or not 0 <= value <= policy["max_teacher_score"]:
            return result
    for observation in (row, verification):
        if transcript_matches(observation["transcript"], protected_phrases):
            return {"eligible": False, "reason": "wake_word_or_disagreement"}
        if not _confident(observation, policy["min_word_probability"]):
            return result
    for phrase in policy["known_negative_phrases"]:
        if tokens(texts[0]) == tokens(phrase):
            return {
                "eligible": True,
                "reason": "known_negative_teacher_stt_consensus",
                "phrase": phrase,
                "teacher_hashes": [t["sha256"] for t in teachers],
                "teacher_calibrations": {t["sha256"]: policy["teacher_calibrations"][t["sha256"]] for t in teachers},
            }
    return result
