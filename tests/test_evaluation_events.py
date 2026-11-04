"""Hand-traced feature/output sequences; no models, TensorFlow or audio data."""

from dataclasses import asdict

import numpy as np
import pytest

from wake_word_trainer.evaluate import Report, count_activations, false_accepts_per_hour
from wake_word_trainer.project import ProjectError


def events(codes, *, cutoff=128, window=1, stride=1, feature_count=None):
    return count_activations(
        np.asarray(codes) / 255,
        np.array([cutoff]) / 255,
        window=window,
        stride=stride,
        feature_count=feature_count,
    )[0]


def test_startup_requires_100_strictly_lower_features():
    assert events([0] * 99 + [255]) == 0
    assert events([0] * 100 + [255]) == 1
    assert events([128] * 100 + [255]) == 0  # equality does not rearm
    assert events([0] * 100 + [128]) == 0  # equality does not detect


def test_long_high_plateau_is_one_event_not_periodic_cooldown_events():
    assert events([0] * 100 + [255] * 500) == 1
    assert events([255] * 500) == 0  # startup has not cooled off either


def test_rearm_is_cumulative_not_consecutive_and_restarts_after_detection():
    first = [0] * 100 + [255]
    assert events(first + [0] * 49 + [255] + [0] * 50 + [255]) == 1
    assert events(first + [0] * 49 + [255] + [0] * 51 + [255]) == 2


def test_integer_window_sum_strict_boundary():
    prefix = [0] * 100
    # 127 + 129 = 128 * 2; a single additional code, not rounding, wins.
    assert events([*prefix, 127, 129], window=2) == 0
    assert events([*prefix, 127, 130], window=2) == 1
    # The initial window is zero-filled; no need to wait for five HIGH outputs.
    assert events(prefix + [255] * 3, cutoff=200, window=5) == 0
    assert events(prefix + [255] * 4, cutoff=200, window=5) == 1


def test_reset_zero_is_used_between_invocations_and_stride_is_not_seconds():
    # After a hit the current slot is zero. At stride 100 there are only 99
    # low feature steps before the next high output; stride 101 allows 100.
    assert events([0, 255, 255, 255], stride=100) == 1
    assert events([0, 255, 255, 255], stride=101) == 3


def test_ignored_output_can_detect_on_a_trailing_feature_without_new_inference():
    # First hit at feature 105. 96 low steps, then 2 stale-zero steps leave
    # two to rearm. At feature 210 the window [255,255,0] exceeds 128*3,
    # but one low step is still missing. Feature 211 releases that pending
    # output even though the next inference would not arrive until 213.
    codes = [0] * 33 + [255] * 2 + [0] * 32 + [255, 255, 0]
    assert len(codes) * 3 == 210
    assert events(codes, window=3, stride=3) == 1
    assert events(codes, window=3, stride=3, feature_count=211) == 2
    assert events(codes, window=3, stride=3, feature_count=212) == 2


def test_cutoff_zero_never_rearms_and_counts_need_not_be_monotonic():
    # Cannot stop evaluating higher thresholds just because the first has no
    # events. Also accept deliberately unordered thresholds without pruning.
    result = count_activations(
        np.asarray([0] * 100 + [255]) / 255,
        np.array([0, 255, 128]) / 255,
        window=1,
        stride=1,
    )
    np.testing.assert_array_equal(result, [0, 0, 1])


def test_rate_includes_startup_and_window_time_and_resets_each_track():
    track = np.asarray([0] * 100 + [255] * 4) / 255
    rates, hours = false_accepts_per_hour([track, track], window=5, stride=1)
    assert hours == pytest.approx(208 * 0.01 / 3600)
    assert rates[200] * hours == pytest.approx(2)
    # No cross-recording warmup leakage in the event counter.
    rates, hours = false_accepts_per_hour([np.zeros(99), np.ones(10)], window=1, stride=1)
    assert rates[128] == 0
    assert hours == pytest.approx(109 * 0.01 / 3600)


def test_rate_counts_trailing_feature_exposure_and_pending_event():
    track = np.asarray([0] * 33 + [255] * 2 + [0] * 32 + [255, 255, 0]) / 255
    rates, hours = false_accepts_per_hour([track], window=3, stride=3, feature_counts=[211])
    assert hours == pytest.approx(2.11 / 3600)
    assert rates[128] * hours == pytest.approx(2)


def test_float_diagnostic_is_not_silently_requantized():
    track = np.r_[np.zeros(100), np.tile([0.498, 0.506], 10)]
    args = dict(cutoffs=np.array([128 / 255]), window=2, stride=1)
    assert count_activations(track, quantized=False, **args)[0] == 1
    assert count_activations(track, quantized=True, **args)[0] == 0


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-0.01], [1.01], [[0.5]]])
def test_invalid_scores_are_not_silent_success(values):
    with pytest.raises(ValueError, match="Probabilities and cutoffs"):
        count_activations(values, window=1, stride=1)


@pytest.mark.parametrize("kwargs", [{"window": 0}, {"stride": 0}, {"feature_count": 2}, {"feature_count": 6}])
def test_invalid_frame_contract(kwargs):
    with pytest.raises(ValueError):
        count_activations([0.0], **({"window": 1, "stride": 3} | kwargs))


def test_missing_or_extra_track_lengths_and_no_exposure_fail():
    for lengths in ([], [3, 3]):
        with pytest.raises(ValueError, match="feature count"):
            false_accepts_per_hour([np.zeros(1)], window=1, stride=3, feature_counts=lengths)
    with pytest.raises(ProjectError, match="No background audio"):
        false_accepts_per_hour([], window=1, stride=3)


def test_old_reports_do_not_acquire_new_event_policy_on_read():
    old = asdict(Report("Hey Nova", "model", 1, "own", 1.0, 5, 0.1, 0.9, 1.0, 0.0))
    old.pop("ambient_event_counting")
    assert Report(**old).ambient_event_counting is None
