"""Portabler Import, getrennte Feature-Räume und ein vollständiger kleiner Teacher-Lauf."""

import json

import numpy as np
import pytest
import yaml

from wake_word_trainer import audio
from wake_word_trainer.project import Project, ProjectError
from wake_word_trainer.teacher import (
    Frontend,
    Parameters,
    Teacher,
    import_checkpoint,
    network,
    source_balance,
    train,
    training_sources,
    training_window,
)

torch = pytest.importorskip("torch")


def test_legacy_import_preserves_logits_and_no_longer_needs_source(tmp_path):
    torch.manual_seed(1)
    model = network().eval()
    original = tmp_path / "old.pt"
    torch.save({"model_state_dict": model.state_dict()}, original)
    config = tmp_path / "old.yaml"
    config.write_text(yaml.safe_dump({"features": {"sample_rate": 16000}, "data": {"negative_mmap_roots": []}}))
    target = tmp_path / "new.pt"
    import_checkpoint(original, config, target)
    original.unlink()
    config.unlink()
    teacher = Teacher(target, device="cpu")
    samples = np.sin(np.arange(48000) * 0.1).astype(np.float32) * 0.1
    features = Frontend(Parameters()).windows(samples)
    with torch.inference_mode():
        expected = torch.sigmoid(model(torch.from_numpy(features).unsqueeze(1))).numpy().reshape(-1)
    actual = teacher(samples)
    assert actual["peak"] == pytest.approx(float(expected.max()), abs=1e-7)
    assert actual["p95"] == pytest.approx(float(np.quantile(expected, 0.95)), abs=1e-7)
    assert actual["feature_space"] == "logmel"


def test_cannot_import_teacher_with_mixed_microfrontend_negatives(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("data:\n  negative_mmap_roots: [microfrontend]\n")
    with pytest.raises(ProjectError, match="gemischten Feature"):
        import_checkpoint(tmp_path / "absent.pt", config, tmp_path / "result.pt")


@pytest.mark.parametrize("architecture", ["legacy_v2", "temporal_v1"])
def test_tiny_training_and_scoring_with_stable_eval_split(tmp_path, architecture):
    project = Project.create(tmp_path / "project", "Hey Nova", "de")
    for split in ("train", "eval"):
        for index in range(2):
            samples = np.sin(np.arange(24000) * (0.05 + index * 0.03)).astype(np.float32) * 0.2
            samples[: index + (1 if split == "eval" else 3)] = 0
            audio.write(project.recordings / split / "room" / f"p{index}.wav", samples)
            audio.write(project.recordings / "negative" / split / "room" / f"n{index}.wav", samples * 0.01)
    checkpoint = train(project, epochs=1, batch_size=2, device="cpu", balanced_sources=True, architecture=architecture)
    assert checkpoint.is_file()
    assert torch.load(checkpoint, weights_only=True)["balanced_sources"] is True
    assert Teacher(checkpoint, device="cpu").metadata["architecture"] == architecture
    assert 0 <= Teacher(checkpoint, device="cpu")(np.ones(24000, np.float32) * 0.1)["peak"] <= 1
    assert len(json.loads(project.path("teacher", "training.json").read_text())) == 1


def test_teacher_refuses_train_eval_leakage(tmp_path):
    project = Project.create(tmp_path / "project", "Hey Nova", "de")
    for split in ("train", "eval"):
        audio.write(project.recordings / split / "room" / "p.wav", np.ones(24000) * 0.1)
        audio.write(project.recordings / "negative" / split / "room" / "n.wav", np.ones(24000) * 0.01)
    with pytest.raises(ProjectError, match="identischer Inhalt"):
        train(project, epochs=1, device="cpu")


def test_score_cli_keeps_audio_hash_separate_from_model_hash(tmp_path, monkeypatch):
    from wake_word_trainer.cli import main
    from wake_word_trainer.datasets import digest

    project = Project.create(tmp_path / "p", "Hey Nova")
    clip = project.recordings / "train" / "room" / "p.wav"
    audio.write(clip, np.ones(24000) * 0.1)

    class FakeTeacher:
        def __init__(self, *args, **kwargs):
            self.metadata = {"sha256": "model-hash"}

        def __call__(self, samples):
            return {"sha256": "model-hash", "peak": 0.9}

    monkeypatch.setattr("wake_word_trainer.teacher.Teacher", FakeTeacher)
    assert main(["-p", str(project.root), "teacher", "score"]) == 0
    result = json.loads(project.path("teacher/scores.json").read_text())
    assert result["clips"][0]["sha256"] == digest(clip)
    assert result["clips"][0]["teacher"]["sha256"] == "model-hash"


def test_corpus_teacher_keeps_external_holdout_and_filters_short_tts(tmp_path):
    from wake_word_trainer.features import _holdout

    project = Project.create(tmp_path / "project", "Hey Nova", "de")
    project.config["negatives"].update(speech_folders=[str(tmp_path / "speech")], eval_share=0.2)
    project.config["tts"]["min_positive_duration_seconds"] = 0.6
    for index in range(10):
        samples = np.random.default_rng(index).normal(0, 0.02, 16000).astype(np.float32)
        audio.write(tmp_path / "speech" / f"{index}.wav", samples)
    audio.write(project.path("tts", "positive", "short.wav"), np.zeros(4000))
    audio.write(project.path("tts", "positive", "long.wav"), np.ones(16000) * 0.1)
    audio.write(project.path("tts", "negative", "similar.wav"), np.ones(16000) * 0.01)
    sources = training_sources(project, True, 3)
    _, validation, testing = _holdout(audio.wavs(tmp_path / "speech"), 0.2)
    assert not set(sources["train_negative"]) & set(validation + testing)
    assert len(sources["train_negative"]) == 4  # Drei externe plus ein ähnliches TTS-Wort.
    assert [p.name for p in sources["train_positive"]] == ["long.wav"]


def test_corpus_teacher_checks_tts_against_own_eval_before_training(tmp_path):
    project = Project.create(tmp_path / "project", "Hey Nova")
    samples = np.ones(16000) * 0.1
    audio.write(project.path("tts", "positive", "same.wav"), samples)
    audio.write(project.recordings / "eval" / "room" / "same.wav", samples)
    with pytest.raises(ProjectError, match="Training/Bewertung"):
        training_sources(project, True)


@pytest.mark.parametrize("length,hop", [(9000, 0.1), (48371, 0.1), (48777, 0.105)])
def test_shared_stft_keeps_all_original_teacher_windows(length, hop):
    frontend = Frontend(Parameters())
    samples = np.random.default_rng(7).normal(0, 0.05, length).astype(np.float32)
    padded = np.pad(samples, (0, max(0, frontend.length - len(samples))))
    last = len(padded) - frontend.length
    starts = list(range(0, last + 1, max(1, round(hop * audio.RATE))))
    if starts[-1] != last:
        starts.append(last)
    expected = np.stack([frontend.clip(padded[s:s + frontend.length]) for s in starts])
    actual = frontend.windows(samples, hop)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-6)


def test_large_corpus_cannot_dilute_satellite_examples_with_source_balancing(tmp_path):
    p = Project.create(tmp_path / "p", "Hey Nova")
    for size in (10, 1000):
        sources = {
            "train_positive": [p.recordings / "train" / "satellite" / "p.wav"],
            "train_negative": [p.recordings / "negative" / "train" / "satellite" / "n.wav"]
                              + [p.path("tts", "negative", f"{i}.wav") for i in range(size)],
        }
        weights, ratio, groups = source_balance(p, sources)
        assert weights[1] / weights[1:].sum() == pytest.approx(4 / 7)
        assert weights[0] * ratio == pytest.approx(weights[1:].sum())
        assert groups["satellite_negative"] == 1


def test_teacher_keeps_the_trigger_at_the_end_of_owned_negative_recordings(tmp_path):
    p = Project.create(tmp_path / "p", "Hey Nova")
    samples = np.zeros(48000, np.float32)
    samples[-6000:] = 0.2
    for folder in (p.recordings / "negative" / "train" / "satellite", p.path("tts", "negative")):
        path = folder / "negative.wav"
        audio.write(path, samples)
        for seed in (1, 9, 123):
            actual = training_window(p, path, samples, 24000, seed, "train", 0)
            np.testing.assert_array_equal(actual, samples[-24000:])
    external = tmp_path / "speech.wav"
    audio.write(external, samples)
    first = training_window(p, external, samples, 24000, 1, "train", 0)
    np.testing.assert_array_equal(first, training_window(p, external, samples, 24000, 1, "train", 0))
    assert first.shape == (24000,)
