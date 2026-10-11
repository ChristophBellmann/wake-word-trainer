"""Compare with the deployed model and run a locally configured firmware rollout.

The HTTP caller cannot choose paths or commands. Training completion and
firmware deployment have separate states, so a failed OTA never erases a
successful training. Runtime state and reference models stay in the project.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .project import Project, ProjectError
from .state import now


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision(candidate: dict, reference: dict, settings: dict) -> tuple[bool, str]:
    """Require improvement at the actual deployed cutoff within the local budget."""
    if not candidate.get("positives") or candidate.get("positive_source") != "own":
        return False, "No held-out own recordings"
    if candidate.get("positives") != reference.get("positives"):
        return False, "Different evaluation sets"
    recall, old_recall = candidate.get("recall"), reference.get("recall")
    faph = candidate.get("false_accepts_per_hour")
    if recall is None or not recall or old_recall is None or faph is None:
        return False, "Incomplete evaluation"
    if faph > float(settings["max_false_accepts_per_hour"]):
        return False, "False activation budget exceeded"
    negatives = candidate.get("own_negatives", 0)
    triggered = candidate.get("own_negatives_triggered", 0)
    old_triggered = reference.get("own_negatives_triggered", 0)
    if negatives != reference.get("own_negatives", 0):
        return False, "Different own negative evaluation sets"
    if negatives and triggered / negatives > float(settings["max_own_negative_share"]):
        return False, "Own negative budget exceeded"
    if recall < old_recall or triggered > old_triggered:
        return False, "Regression against the deployed model"
    satellite = candidate.get("satellite_negatives_triggered")
    old_satellite = reference.get("satellite_negatives_triggered")
    if satellite is not None and old_satellite is not None and satellite > old_satellite:
        return False, "More satellite false activations than the deployed model"
    if recall <= old_recall and triggered >= old_triggered:
        return False, "No measured improvement"
    return True, "Improved against the deployed model within the false activation budget"


def assess(project: Project, downloads: Path) -> dict:
    from .evaluate import compare

    config = project.config["deployment"]
    candidate = project.export_dir / f"{project.slug}.tflite"
    reference = project.resolve(config["reference_model"])
    if not reference.is_file() or not reference.with_suffix(".json").is_file():
        raise ProjectError("deployment.reference_model needs the deployed model and its manifest")
    candidate_hash, reference_hash = digest(candidate), digest(reference)
    if candidate_hash == reference_hash:
        result = {"passed": False, "reason": "This model is already deployed"}
    else:
        rows = compare(project, [candidate, reference], downloads)
        new = rows[0].get("own_cutoff", rows[0]["budget"])
        old = rows[1].get("own_cutoff", rows[1]["budget"])
        passed, reason = decision(new, old, project.config["evaluation"])
        result = {"passed": passed, "reason": reason, "candidate": new, "reference": old}
    result.update(
        candidate_sha256=candidate_hash,
        reference_sha256=reference_hash,
        candidate_manifest_sha256=digest(candidate.with_suffix(".json")),
        reference_manifest_sha256=digest(reference.with_suffix(".json")),
        checked_at=now(),
    )
    report_path = project.path("report.json")
    report = json.loads(report_path.read_text())
    report["parity"] = result
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    project.path("parity.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def record(project: Project, state: str, **values) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=project.root, delete=False, encoding="utf-8") as stream:
        stream.write(json.dumps({"state": state, "updated_at": now(), **values}, indent=2) + "\n")
        temporary = stream.name
    os.replace(temporary, project.path("deployment.json"))


def deploy(project: Project) -> bool:
    try:
        return _deploy(project)
    except Exception as err:
        record(project, "failed", reason=f"Deployment failed ({type(err).__name__}); see deployment.log")
        return False


def _deploy(project: Project) -> bool:
    config = project.config["deployment"]
    parity = json.loads(project.path("parity.json").read_text())
    candidate = project.export_dir / f"{project.slug}.tflite"
    reference = project.resolve(config["reference_model"])
    if (
        not parity.get("passed")
        or parity.get("candidate_sha256") != digest(candidate)
        or parity.get("candidate_manifest_sha256") != digest(candidate.with_suffix(".json"))
    ):
        record(project, "blocked", reason="No passed comparison for the exported model")
        return False
    if parity.get("reference_sha256") != digest(reference) or parity.get("reference_manifest_sha256") != digest(
        reference.with_suffix(".json")
    ):
        record(project, "blocked", reason="The deployed reference has changed; compare again")
        return False
    command = [
        part.replace("{sha256}", digest(candidate))
        .replace("{export}", str(project.export_dir))
        .replace("{parity}", str(project.path("parity.json")))
        for part in config["command"]
    ]
    record(project, "running", model_sha256=digest(candidate), started_at=now(), pid=os.getpid())
    # Do not capture commands or credentials in HTTP output. Detailed output is local only.
    with project.path("deployment.log").open("a") as log:
        result = subprocess.run(command, cwd=project.root, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        record(
            project,
            "failed",
            model_sha256=digest(candidate),
            exit_code=result.returncode,
            reason="Firmware rollout failed; see deployment.log and retry deployment",
        )
        return False
    reference.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate, reference)
    shutil.copyfile(candidate.with_suffix(".json"), reference.with_suffix(".json"))
    record(project, "completed", model_sha256=digest(candidate), ended_at=now())
    return True


def main() -> int:
    project = Project.load(sys.argv[1])
    try:
        return 0 if deploy(project) else 1
    except Exception as err:
        record(project, "failed", reason=f"Deployment failed ({type(err).__name__}); see deployment.log")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
