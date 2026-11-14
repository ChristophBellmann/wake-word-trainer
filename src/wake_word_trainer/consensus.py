"""Begrenzte Negativfreigabe: zwei STT und geprüfter Teacher; bekannte Phrase oder kein Aktivierungswort."""

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
        "no_wake_word",
        "no_wake_word_devices",
        "protected_phrases",
    }
    if set(policy) - allowed:
        raise ProjectError("Unbekannte Einstellung in Negativrichtlinie")
    phrases = policy.get("known_negative_phrases", [])
    hashes = policy.get("allowed_teacher_hashes", [])
    no_wake_word = policy.get("no_wake_word", False)
    if not isinstance(no_wake_word, bool):
        raise ProjectError("no_wake_word in der Negativrichtlinie muss true oder false sein")
    protected = policy.get("protected_phrases", [])
    if (
        not isinstance(protected, list)
        or len(protected) > 100
        or any(not isinstance(p, str) or not tokens(p) or len(p) > 200 for p in protected)
    ):
        raise ProjectError("Ungültige geschützte Phrasen in Negativrichtlinie")
    devices = policy.get("no_wake_word_devices", [])
    if (
        not isinstance(devices, list)
        or len(devices) > 100
        or any(not isinstance(d, str) or not re.fullmatch("[a-zA-Z0-9_-]{1,64}", d) for d in devices)
        # A teacher is calibrated per microphone: the rule names its devices.
        or bool(devices) != no_wake_word
    ):
        raise ProjectError("no_wake_word benötigt no_wake_word_devices mit den kalibrierten Collector-Geräten")
    if (
        not isinstance(phrases, list)
        # Without the explicit no-wake-word rule a policy needs known phrases.
        or not (0 if no_wake_word else 1) <= len(phrases) <= 100
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
    result = {
        "min_word_probability": 0.85,
        "max_teacher_score": 0.1,
        "teacher_calibrations": {},
        "known_negative_phrases": [],
        "no_wake_word": False,
        "no_wake_word_devices": [],
        "protected_phrases": [],
        **policy,
    }
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


NO_WAKE_WORD = "no_wake_word_teacher_stt_consensus"
KNOWN_NEGATIVE = "known_negative_teacher_stt_consensus"
REASONS = {KNOWN_NEGATIVE, NO_WAKE_WORD}


def _teachers(row: dict, positive_phrases: list[str], policy: dict) -> tuple[dict | None, list[str]]:
    """Every teacher must be approved, calibrated for the project targets and score low."""
    unclear = {"eligible": False, "reason": "requires_review"}
    teachers = row.get("teachers")
    if not isinstance(teachers, list) or not teachers:
        return unclear, []
    project_targets = _phrases(positive_phrases)
    if not project_targets:
        return {"eligible": False, "reason": "teacher_target_or_calibration_unclear"}, []
    protected = list(positive_phrases)
    for teacher in teachers:
        if not isinstance(teacher, dict) or teacher.get("error"):
            return unclear, []
        digest = teacher.get("sha256")
        if not isinstance(digest, str) or digest not in policy["allowed_teacher_hashes"]:
            return unclear, []
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
            return {"eligible": False, "reason": "teacher_target_or_calibration_unclear"}, []
        protected.extend(teacher["target_phrases"])
        value = teacher.get("peak")
        if not _number(value) or not 0 <= value <= policy["max_teacher_score"]:
            return unclear, []
    return None, protected


def _no_wake_word(row: dict, positive_phrases: list[str], policy: dict) -> dict:
    """Two different STT models hear no wake word and every calibrated teacher scores low.

    The transcripts may be empty or differ: Whisper misses most real wake
    words, so its text never proves a false alarm. It only vetoes. The
    acoustic teacher carries the decision, and only for the devices whose
    microphone it was checked on: a teacher that separates one room can score
    real wake words near zero in another.
    """
    result = {"eligible": False, "reason": "requires_review"}
    if row.get("error") or row.get("category") != "triggers":
        return result
    devices = policy.get("no_wake_word_devices")
    if not isinstance(devices, list) or not isinstance(row.get("device"), str) or row["device"] not in devices:
        return {"eligible": False, "reason": "device_not_calibrated"}
    verification = row.get("verification")
    if not isinstance(verification, dict) or verification.get("error"):
        return result
    for observation in (row, verification):
        stt = observation.get("stt")
        if not isinstance(stt, dict) or stt.get("error") or not isinstance(stt.get("backend"), str):
            return result
        if not isinstance(observation.get("transcript"), str):
            return result
    blocked, protected = _teachers(row, positive_phrases, policy)
    if blocked:
        return blocked
    protected += policy.get("protected_phrases", [])
    for observation in (row, verification):
        if transcript_matches(observation["transcript"], protected):
            return {"eligible": False, "reason": "wake_word_or_disagreement"}
    teachers = row["teachers"]
    return {
        "eligible": True,
        "reason": NO_WAKE_WORD,
        "teacher_hashes": [t["sha256"] for t in teachers],
        "teacher_calibrations": {t["sha256"]: policy["teacher_calibrations"][t["sha256"]] for t in teachers},
    }


def decide(row: dict, positive_phrases: list[str], policy: dict) -> dict:
    if row.get("origin_ambiguous"):
        return {"eligible": False, "reason": "ambiguous_origin"}
    known = _known_phrase(row, positive_phrases, policy)
    if known["eligible"] or policy.get("no_wake_word") is not True:
        return known
    return _no_wake_word(row, positive_phrases, policy)


def _known_phrase(row: dict, positive_phrases: list[str], policy: dict) -> dict:
    result = {"eligible": False, "reason": "requires_review"}
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
    blocked, protected_phrases = _teachers(row, positive_phrases, policy)
    if blocked:
        return blocked
    teachers = row["teachers"]
    for observation in (row, verification):
        if transcript_matches(observation["transcript"], protected_phrases):
            return {"eligible": False, "reason": "wake_word_or_disagreement"}
        if not _confident(observation, policy["min_word_probability"]):
            return result
    for phrase in policy["known_negative_phrases"]:
        if tokens(texts[0]) == tokens(phrase):
            return {
                "eligible": True,
                "reason": KNOWN_NEGATIVE,
                "phrase": phrase,
                "teacher_hashes": [t["sha256"] for t in teachers],
                "teacher_calibrations": {t["sha256"]: policy["teacher_calibrations"][t["sha256"]] for t in teachers},
            }
    return result
