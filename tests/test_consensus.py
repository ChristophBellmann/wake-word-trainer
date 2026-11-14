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


# No-wake-word rule: the teacher decides, both STT models may only veto. ---------

NO_WAKE_WORD_POLICY = {
    **POLICY,
    "known_negative_phrases": [],
    "no_wake_word": True,
    "no_wake_word_devices": ["bedroom"],
    "protected_phrases": ["hey nowa"],
}


def silent_trigger():
    """A throat-clearing style trigger: no words, two CPU STT models, low teacher score."""
    return {
        "category": "triggers",
        "device": "bedroom",
        "transcript": "",
        "words": [],
        "stt": {"backend": "faster_whisper", "segments": []},
        "teachers": [{"sha256": HASH, "peak": 0.03, "target_phrases": ["hey nova"]}],
        "verification": {"transcript": "", "words": [], "stt": {"backend": "faster_whisper", "segments": []}},
    }


def test_no_wake_word_rule_accepts_silence_and_differing_speech_with_low_teacher():
    result = decide(silent_trigger(), ["hey nova"], NO_WAKE_WORD_POLICY)
    assert result["eligible"] is True
    assert result["reason"] == "no_wake_word_teacher_stt_consensus"
    assert result["teacher_hashes"] == [HASH]
    row = silent_trigger()
    row["transcript"], row["verification"]["transcript"] = "The instructions or files.", "Thanks a lot."
    assert decide(row, ["hey nova"], NO_WAKE_WORD_POLICY)["eligible"] is True


def test_no_wake_word_rule_is_off_unless_the_policy_enables_it():
    assert decide(silent_trigger(), ["hey nova"], POLICY)["eligible"] is False
    assert decide(silent_trigger(), ["hey nova"], {**NO_WAKE_WORD_POLICY, "no_wake_word": "yes"})["eligible"] is False


def test_known_phrase_rule_keeps_working_next_to_the_no_wake_word_rule():
    policy = {**POLICY, "no_wake_word": True, "no_wake_word_devices": ["bedroom"], "protected_phrases": []}
    result = decide(observation(), ["hey nova"], policy)
    assert result["eligible"] is True and result["reason"] == "known_negative_teacher_stt_consensus"


@pytest.mark.parametrize("side", ["primary", "verification"])
@pytest.mark.parametrize("text", ["Hey Nova!", "well, hey nova please", "Hey Nowa.", "heynova"])
def test_either_stt_hearing_a_wake_word_or_protected_spelling_vetoes(side, text):
    row = silent_trigger()
    (row if side == "primary" else row["verification"])["transcript"] = text
    assert decide(row, ["hey nova"], NO_WAKE_WORD_POLICY) == {"eligible": False, "reason": "wake_word_or_disagreement"}


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row["teachers"][0].update(peak=0.11),
        lambda row: row["teachers"][0].update(peak=True),
        lambda row: row["teachers"][0].update(peak=float("nan")),
        lambda row: row["teachers"][0].update(sha256="b" * 64),
        lambda row: row["teachers"][0].update(target_phrases=[]),
        lambda row: row["teachers"][0].update(error="synthetic failure"),
        lambda row: row.update(teachers=[]),
        lambda row: row["teachers"].append({"sha256": HASH, "peak": 0.9, "target_phrases": ["hey nova"]}),
        lambda row: row.update(category="needs_review"),
        lambda row: row.update(device="kitchen"),
        lambda row: row.update(device=None),
        lambda row: row.pop("device"),
        lambda row: row.update(error="synthetic failure"),
        lambda row: row.update(origin_ambiguous=True),
        lambda row: row.update(transcript=None),
        lambda row: row.pop("verification"),
        lambda row: row["verification"].update(error="synthetic failure"),
        lambda row: row["verification"].update(transcript=None),
        lambda row: row["stt"].update(error="synthetic failure"),
        lambda row: row["verification"]["stt"].update(error="synthetic failure"),
        lambda row: row.update(stt={}),
        lambda row: row["verification"].update(stt=None),
    ],
)
def test_no_wake_word_rule_fails_closed(change):
    row = silent_trigger()
    change(row)
    assert decide(row, ["hey nova"], NO_WAKE_WORD_POLICY)["eligible"] is False


def test_no_wake_word_rule_needs_the_same_teacher_calibration_contract():
    policy = deepcopy(NO_WAKE_WORD_POLICY)
    policy["teacher_calibrations"] = {}
    result = decide(silent_trigger(), ["hey nova"], policy)
    assert result == {"eligible": False, "reason": "teacher_target_or_calibration_unclear"}
    # A project target the teacher was not trained for stays unclear.
    result = decide(silent_trigger(), ["hey nova", "hi nova"], NO_WAKE_WORD_POLICY)
    assert result == {"eligible": False, "reason": "teacher_target_or_calibration_unclear"}


def test_policy_may_omit_known_phrases_only_with_the_no_wake_word_rule(tmp_path):
    import yaml

    path = tmp_path / "policy.yaml"
    settings = deepcopy(NO_WAKE_WORD_POLICY)
    settings.pop("known_negative_phrases")
    path.write_text(yaml.safe_dump(settings))
    loaded = load_policy(path)
    assert loaded["no_wake_word"] is True and loaded["known_negative_phrases"] == []
    assert decide(silent_trigger(), ["hey nova"], loaded)["eligible"] is True
    for broken in (
        {**settings, "no_wake_word": False},
        {**settings, "no_wake_word": "true"},
        {**settings, "no_wake_word_devices": []},
        {**settings, "no_wake_word_devices": "bedroom"},
        {**settings, "no_wake_word_devices": ["../bedroom"]},
        {**{k: v for k, v in settings.items() if k != "no_wake_word_devices"}},
        {**POLICY, "no_wake_word_devices": ["bedroom"]},
        {**settings, "protected_phrases": "hey nowa"},
        {**settings, "protected_phrases": [""]},
        {**settings, "protected_phrases": [7]},
    ):
        path.write_text(yaml.safe_dump(broken))
        with pytest.raises(ProjectError):
            load_policy(path)


def test_a_device_without_calibration_is_never_classified_by_the_no_wake_word_rule():
    row = silent_trigger()
    row["device"] = "kitchen"
    assert decide(row, ["hey nova"], NO_WAKE_WORD_POLICY) == {"eligible": False, "reason": "device_not_calibrated"}
