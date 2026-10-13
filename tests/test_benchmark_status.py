from wake_word_trainer.cli import main
from wake_word_trainer.project import Project
from wake_word_trainer.state import State, read


def test_benchmark_completion_cannot_offer_previous_model_to_collector(tmp_path, monkeypatch):
    project = Project.create(tmp_path / "p", "Hey Nova")
    project.export_dir.mkdir()
    model = project.export_dir / f"{project.slug}.tflite"
    model.write_bytes(b"previous-model")
    State(project).update(state="completed", best_model_available=True, ended_at="old-training")

    def benchmark(*args, **kwargs):
        assert read(project)["best_model_available"] is False
        return {"variants": [], "failed_runs": 0}

    monkeypatch.setattr("wake_word_trainer.benchmark.run", benchmark)
    assert main(["-p", str(project.root), "benchmark"]) == 0
    status = read(project)
    assert status["state"] == "completed"
    assert status["best_model_available"] is False
    assert status["ended_at"] != "old-training"
    assert model.read_bytes() == b"previous-model"
