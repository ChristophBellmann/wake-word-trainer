"""GPU-Belegung, skalierungstreue Graph-Schritte, Cache-Gates und vollständige Bewertung."""

import json

import numpy as np
import pytest

from wake_word_trainer.gpu import busy, environment, lease, snapshot
from wake_word_trainer.project import Project, ProjectError


def test_gpu_lease_is_exclusive_and_releases_after_failure(tmp_path):
    path = tmp_path / "gpu.lock"
    with pytest.raises(RuntimeError), lease("first", path):
        assert busy(path)
        with pytest.raises(ProjectError, match="bereits"):
            with lease("second", path):
                pass
        raise RuntimeError("interrupted")
    assert not busy(path)
    with lease("next", path):
        assert json.loads(path.read_text())["job"] == "next"


def test_runtime_environment_preserves_explicit_settings(tmp_path):
    root = tmp_path / "rocm"
    (root / "lib").mkdir(parents=True)
    (root / "bin").mkdir()
    original = {
        "ROCM_PATH": str(root),
        "PATH": "/custom/bin",
        "TF_ROCM_DISABLE_HIPBLASLT": "0",
        "LD_LIBRARY_PATH": "/custom/lib",
    }
    result = environment(original)
    assert result["TF_ROCM_DISABLE_HIPBLASLT"] == "0"
    assert result["PATH"].endswith("/custom/bin") and result["LD_LIBRARY_PATH"].endswith("/custom/lib")
    assert original["PATH"] == "/custom/bin"


def test_amd_telemetry_does_not_need_rocm_smi(tmp_path):
    root = tmp_path / "card1/device"
    root.mkdir(parents=True)
    for name, value in {"mem_info_vram_total": 1000, "mem_info_vram_used": 250, "gpu_busy_percent": 83}.items():
        (root / name).write_text(str(value))
    assert snapshot(tmp_path) == {"use_percent": 83.0, "memory_percent": 25.0, "vram_used_bytes": 250}


def test_service_refuses_gpu_job_before_touching_state(tmp_path, monkeypatch):
    from wake_word_trainer.service import Service

    project = Project.create(tmp_path / "p", "Hey Nova")
    token = tmp_path / "token"
    token.write_text("test-token-0123456789")
    service = Service({"project": str(project.root), "token_file": str(token)})
    monkeypatch.setattr("wake_word_trainer.gpu.busy", lambda: True)
    code, result = service.start(next(iter(service.profiles)))
    assert code == 409 and "GPU" in result["error"]
    assert not project.path("state.json").exists()


def test_successful_benchmark_cannot_roll_out_a_previous_ready_model(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from wake_word_trainer.service import Service
    from wake_word_trainer.state import State

    project = Project.create(
        tmp_path / "p", "Hey Nova", deployment={"enabled": True, "reference_model": "ref.tflite", "command": ["false"]}
    )
    token = tmp_path / "token"
    token.write_text("test-token-0123456789")
    service = Service({"project": str(project.root), "token_file": str(token)})
    process = SimpleNamespace(wait=lambda: 0, returncode=0)
    service.process = process
    State(project).update(best_model_available=True)
    project.path("deployment.json").write_text('{"state":"ready"}')

    def forbidden(*args, **kwargs):
        raise AssertionError("Benchmark darf keinen Rollout starten")

    monkeypatch.setattr("wake_word_trainer.service.subprocess.Popen", forbidden)
    service._finished(process, allow_deployment=False)
    assert service.deployment_process is None


@pytest.mark.slow
def test_graph_step_matches_keras_weighted_loss_and_parameter_update():
    tf = pytest.importorskip("tensorflow")
    from wake_word_trainer.optimized_tf import compile_model, make_step

    def model():
        return tf.keras.Sequential(
            [
                tf.keras.Input((4, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="ones"),
            ]
        )

    first, second = model(), model()
    compile_model(first)
    compile_model(second)
    config = {
        "positive_class_weight": [1.0],
        "negative_class_weight": [20.0],
        "time_mask_count": [0],
        "freq_mask_count": [0],
    }
    x = np.arange(32, dtype=np.float32).reshape(4, 4, 2) / 100
    y = np.array([[1], [0], [1], [0]], np.float32)
    w = np.array([[1], [0.5], [2], [1]], np.float32)
    weighted = w * np.where(y >= 0.5, 1.0, 20.0)
    expected_loss = tf.keras.losses.BinaryCrossentropy()(y, first(x), sample_weight=weighted)
    make_step(first, config, "standard")(x, y, w)
    optimized = make_step(second, config)(x, y, w)
    assert float(optimized.numpy()) == pytest.approx(float(expected_loss.numpy()), rel=1e-5)
    for expected, actual in zip(first.get_weights(), second.get_weights(), strict=True):
        np.testing.assert_allclose(expected, actual, atol=1e-6)
    # Adam-Momente decken auch abweichende Loss-Skalierung auf, die beim ersten
    # Schritt wegen Adams Normalisierung in den Gewichten allein unsichtbar wäre.
    for expected, actual in zip(first.optimizer.variables, second.optimizer.variables, strict=True):
        np.testing.assert_allclose(expected.numpy(), actual.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.slow
def test_streaming_validation_matches_backend_on_every_clip_and_ambient_window(tmp_path):
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("microwakeword")
    from microwakeword.data import FeatureHandler
    from microwakeword.train import validate_nonstreaming

    from wake_word_trainer.features import write_mmap
    from wake_word_trainer.optimized_tf import compile_model
    from wake_word_trainer.validation import StreamingValidation

    config = {
        "stride": 3,
        "window_step_ms": 10,
        "spectrogram_length": 4,
        "training_input_shape": (4, 40),
        "batch_size": 2,
        "validation_batch_size": 3,
        "validation_cache_mb": 2,
        "features": [],
    }
    for name, truth in [("positive", True), ("negative", False)]:
        root = tmp_path / name
        write_mmap(
            root / "validation" / f"{name}_mmap",
            iter([np.ones((4, 40), np.float32) * (0.1 if truth else 0.01) for _ in range(5)]),
        )
        config["features"].append(
            {
                "type": "mmap",
                "features_dir": str(root),
                "truth": truth,
                "sampling_weight": 1,
                "penalty_weight": 1,
                "truncation_strategy": "truncate_start",
            }
        )
    # Lange Spuren erzwingen denselben Split-Fensterpfad wie echte Ambient-Daten.
    write_mmap(tmp_path / "negative/validation_ambient/ambient_mmap", iter([np.ones((160, 40), np.float32) * 0.01]))
    processor = FeatureHandler(config)
    model = tf.keras.Sequential(
        [
            tf.keras.Input((4, 40)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="ones"),
        ]
    )
    compile_model(model)
    expected = validate_nonstreaming(config, processor, model, "validation")
    streamed = StreamingValidation(config, processor, model)
    first = streamed()
    second = streamed()
    for key, value in expected.items():
        assert first[key] == pytest.approx(float(value), rel=1e-5, abs=1e-6), key
        assert second[key] == pytest.approx(first[key], rel=1e-6), key


@pytest.mark.slow
def test_tfrecord_cache_is_unaugmented_reused_and_invalidated_on_sources(tmp_path):
    pytest.importorskip("tensorflow")
    from wake_word_trainer.input_pipeline import tfrecord

    source = tmp_path / "features"
    source.mkdir()
    (source / "data.ninja").write_bytes(b"first")
    config = {
        "features": [{"features_dir": str(source)}],
        "training_input_shape": (4, 2),
        "spectrogram_length": 4,
        "train_dir": str(tmp_path / "model"),
        "seed": 42,
    }

    class Processor:
        def __init__(self):
            self.calls = 0

        def get_data(self, mode, size, length, strategy):
            self.calls += 1
            return np.ones((size, 4, 2), np.float32) * 0.03125, np.ones(size, np.float32), np.ones(size, np.float32)

    processor = Processor()
    path, first = tfrecord(processor, config, 8)
    reused, second = tfrecord(processor, config, 8)
    assert reused == path and second["reused"] and not first["reused"] and processor.calls == 1
    (source / "data.ninja").write_bytes(b"changed")
    updated, third = tfrecord(processor, config, 8)
    assert updated != path and not third["reused"]
