import json
from dataclasses import asdict

import numpy as np

from wake_word_trainer.evaluate import Measurement, report
from wake_word_trainer.project import Project


def test_real_satellite_scores_produce_serializable_report(tmp_path):
    p = Project.create(tmp_path / "p", "Hey Nova")
    positives = [p.recordings / "eval" / "satellite" / "positive.wav"]
    negatives = [p.recordings / "negative" / "eval" / device / "negative.wav"
                 for device in ("satellite_a", "satellite_b", "local")]
    measured = Measurement(p.path("model.tflite"), 5, positives, "own", np.array([0.9]),
                           negatives, np.array([0.9, 0.2, 0.95]), np.zeros(256), 5.0)
    result = report(p, measured, 128, 0.5)
    decoded = json.loads(json.dumps(asdict(result)))
    assert decoded["own_negatives_triggered"] == 2
    assert decoded["satellite_negatives_triggered"] == 1
    assert decoded["satellite_negatives"] == 2
