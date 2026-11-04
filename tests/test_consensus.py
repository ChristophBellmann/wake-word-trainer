"""Unclear audio and model/STT disagreement must never become training labels."""

from copy import deepcopy

import pytest

from wake_word_trainer.consensus import decide, load_policy
from wake_word_trainer.project import ProjectError

HASH = "a" * 64
POLICY = {
    "known_negative_phrases": ["start vacuuming"],
    "allowed_teacher_hashes": [HASH],
    "teacher_calibrations": {
        HASH: {"target_phrases": ["hey nova"], "evidence": "private-review/teacher-validation.json"}
    },
    "min_word_probability": 0.85,
    "max_teacher_score": 0.1,
}


def observation():
    segment = {"avg_logprob": -0.1, "no_speech_prob": 0.01}
    words = [{"word": "start", "probability": 0.98}, {"word": "vacuuming", "probability": 0.97}]
    return {
        "category": "triggers",
        "transcript": "Start vacuuming.",
        "words": words,
        "stt": {"backend": "torch_whisper", "segments": [segment]},
        "teachers": [{"sha256": HASH, "peak": 0.03, "target_phrases": ["hey nova"]}],
        "verification": {
            "transcript": "Start vacuuming.",
            "words": deepcopy(words),
            "stt": {"backend": "faster_whisper", "segments": [deepcopy(segment)]},
        },
    }


def test_only_known_confident_consensus_is_eligible():
    result = decide(observation(), ["hey nova"], POLICY)
    assert result["eligible"] is True
    assert result["teacher_calibrations"] == POLICY["teacher_calibrations"]


def test_full_text_normalization_allows_case_punctuation_and_spacing_only():
    row = observation()
    row["verification"]["transcript"] = "  START, vacuuming! "
    assert decide(row, ["hey nova"], POLICY)["eligible"] is True


def set_text(result, text):
    result["transcript"] = text
    result["words"] = [{"word": word, "probability": 0.99} for word in text.split()]


@pytest.mark.parametrize(
    "first,second",
    [
        ("start vacuuming please", "start vacuuming now"),
        ("do not start vacuuming", "start vacuuming"),
        ("start vacuuming please", "start vacuuming please"),
        ("start vacuuming", "start vacuuming now"),
        ("", ""),
        (None, "start vacuuming"),
    ],
)
def test_shared_partial_phrase_or_extra_words_never_grant_consent(first, second):
    row = observation()
    for result, text in [(row, first), (row["verification"], second)]:
        if isinstance(text, str):
            set_text(result, text)
        else:
            result["transcript"] = text
    assert decide(row, ["hey nova"], POLICY)["eligible"] is False


def test_project_positive_overrides_explicit_negative_phrase():
    row, policy = observation(), deepcopy(POLICY)
    for result in (row, row["verification"]):
        set_text(result, "hey nova")
    policy["known_negative_phrases"] = ["hey nova"]
    assert decide(row, ["hey nova"], policy)["eligible"] is False


def test_teacher_positive_overrides_explicit_negative_phrase_even_with_low_score():
    row, policy = observation(), deepcopy(POLICY)
    targets = ["hey nova", "start vacuuming"]
    row["teachers"][0]["target_phrases"] = targets
    policy["teacher_calibrations"][HASH]["target_phrases"] = targets
    assert decide(row, ["hey nova"], policy)["eligible"] is False


@pytest.mark.parametrize("side", ["primary", "verification"])
@pytest.mark.parametrize("field", ["probability", "avg_logprob", "no_speech_prob"])
@pytest.mark.parametrize("value", [None, True, False, float("nan"), float("inf")])
def test_missing_boolean_or_nonfinite_confidence_is_not_evidence(side, field, value):
    row = observation()
    result = row if side == "primary" else row["verification"]
    target = result["words"][0] if field == "probability" else result["stt"]["segments"][0]
    if value is None:
        target.pop(field)
    else:
        target[field] = value
    assert decide(row, ["hey nova"], POLICY)["eligible"] is False


@pytest.mark.parametrize("value", [None, True, False, float("nan"), float("inf")])
def test_teacher_score_requires_a_real_finite_number(value):
    row = observation()
    row["teachers"][0]["peak"] = value
    assert decide(row, ["hey nova"], POLICY)["eligible"] is False


@pytest.mark.parametrize("side", ["primary", "verification"])
def test_confidences_must_cover_the_whole_transcript(side):
    row = observation()
    result = row if side == "primary" else row["verification"]
    result["words"] = result["words"][:1]
    assert decide(row, ["hey nova"], POLICY)["eligible"] is False


@pytest.mark.parametrize("side", ["primary", "verification"])
@pytest.mark.parametrize("nested", [False, True])
def test_stt_errors_never_grant_consent(side, nested):
    row = observation()
    result = row if side == "primary" else row["verification"]
    (result["stt"] if nested else result)["error"] = "synthetic backend failure"
    assert decide(row, ["hey nova"], POLICY)["eligible"] is False


@pytest.mark.parametrize(
    "change",
    [
        lambda r, p: r["teachers"][0].pop("target_phrases"),
        lambda r, p: r["teachers"][0].update(target_phrases=[]),
        lambda r, p: r["teachers"][0].update(target_phrases=["nova"]),
        lambda r, p: p.pop("teacher_calibrations"),
        lambda r, p: p["teacher_calibrations"].clear(),
        lambda r, p: p["teacher_calibrations"][HASH].update(target_phrases=["nova"]),
        lambda r, p: p["teacher_calibrations"][HASH].update(evidence=""),
    ],
)
def test_target_or_calibration_uncertainty_blocks_a_low_teacher_score(change):
    row, policy = observation(), deepcopy(POLICY)
    change(row, policy)
    assert decide(row, ["hey nova"], policy) == {
        "eligible": False,
        "reason": "teacher_target_or_calibration_unclear",
    }


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(category="candidates"),
        lambda r: r.update(origin_ambiguous=True),
        lambda r: r.update(error="failed"),
        lambda r: r.update(words=[]),
        lambda r: r.update(teachers=[]),
        lambda r: r["teachers"][0].update(peak=0.8),
        lambda r: r["teachers"][0].update(peak=float("nan")),
        lambda r: r["teachers"][0].update(sha256="b" * 64),
        lambda r: r["words"][0].update(probability=0.3),
        lambda r: r["words"][0].update(probability=float("nan")),
        lambda r: r["stt"]["segments"][0].update(avg_logprob=-2),
        lambda r: r["stt"]["segments"][0].update(no_speech_prob=0.99),
        lambda r: r["verification"].update(transcript="Continue moving."),
        lambda r: r["verification"].update(transcript="Start vacuuming. Hey Nova."),
        lambda r: r.update(transcript="Start vacuuming. Hey Nova."),
        lambda r: r["verification"].update(words=[]),
        lambda r: r["teachers"][0].update(error="unavailable"),
        lambda r: r["teachers"][0].update(sha256=[]),
    ],
)
def test_unclear_or_disagreement_is_not_a_negative(change):
    row = observation()
    change(row)
    assert decide(row, ["hey nova"], POLICY)["eligible"] is False


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "known_negative_phrases: []",
        "known_negative_phrases: [start vacuuming]\nallowed_teacher_hashes: [bad]",
        "extra_setting: true",
    ],
)
def test_invalid_policy_fails_closed(tmp_path, content):
    path = tmp_path / "policy.yaml"
    path.write_text(content)
    with pytest.raises(ProjectError):
        load_policy(path)


@pytest.mark.parametrize(
    "calibrations",
    [
        [],
        {HASH: {}},
        {HASH: {"target_phrases": [], "evidence": "review.json"}},
        {HASH: {"target_phrases": ["hey nova"], "evidence": True}},
        {"b" * 64: {"target_phrases": ["hey nova"], "evidence": "review.json"}},
    ],
)
def test_invalid_calibration_contract_fails_closed(tmp_path, calibrations):
    import yaml

    policy = {**POLICY, "teacher_calibrations": calibrations}
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy))
    with pytest.raises(ProjectError):
        load_policy(path)


def test_legacy_hash_allowlist_without_calibration_remains_review_only(tmp_path):
    import yaml

    policy = deepcopy(POLICY)
    policy.pop("teacher_calibrations")
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy))
    assert decide(observation(), ["hey nova"], load_policy(path))["eligible"] is False
