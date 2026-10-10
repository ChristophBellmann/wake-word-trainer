"""HTTP service on the training computer, for Home Assistant (Wake Word Collector).

    GET  /health                        no token
    GET  /v1/status                     progress, best result, GPU, profiles
    POST /v1/start   {"profile": id}    start a run with a profile
    POST /v1/stop                       stop the run (keeps finished rounds)
    GET  /v1/report                     evaluation of the exported model
    GET  /v1/model/<slug>.json|.tflite  the exported model for ESPHome
    POST /v1/speaker_test {"route": r}  play one held-out recording through a
                                        loudspeaker, to test a satellite live

Every /v1 request needs `Authorization: Bearer <token>`. Network callers can
only choose among configured profiles and routes: no commands, paths or
settings reach the service.

Configuration (YAML):

    project: ~/wakeword/hey_jarvis          # project folder with wakeword.yaml
    downloads: ~/wakeword/downloads         # optional, shared downloads
    token_file: ~/.config/wake-word-trainer/service.token
    bind: 0.0.0.0
    port: 10701
    profiles:                               # optional; these are the defaults
      quick: {label: Quick, rounds: 0, steps: 10000}
      recommended: {label: Recommended, rounds: 2}
      thorough: {label: Thorough, rounds: 4, steps: 40000}
      validate: {label: Check data only, prepare_only: true}
    speaker_test:                           # optional
      routes:
        speakers: [paplay, "{file}"]
        usb: [aplay, -D, "plughw:2,0", "{file}"]
    resource_check:                         # optional, see system_check.py
      min_available_memory_gb: 6
      min_free_vram_gb: 8
"""

from __future__ import annotations

import hmac
import importlib.util
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

from . import audio
from .extraction import MAX_BYTES, Extractor
from .project import Project
from .resources import Resources
from .state import State, now, read
from .system_check import SystemCheck

DEFAULT_PROFILES = {
    "quick": {"label": "Quick", "rounds": 0, "steps": 10000},
    "recommended": {"label": "Recommended", "rounds": 2},
    "thorough": {"label": "Thorough", "rounds": 4, "steps": 40000},
    "validate": {"label": "Check data only", "prepare_only": True},
}
MODEL_FILE = re.compile(r"^[a-z0-9_]{1,64}\.(json|tflite)$")


class Service:
    def __init__(self, config: dict[str, Any]):
        self.project = Project.load(config["project"])
        self.downloads = Path(config["downloads"]).expanduser() if config.get("downloads") else None
        token_file = Path(config["token_file"]).expanduser()
        self.token = token_file.read_text(encoding="utf-8").strip()
        if len(self.token) < 16:
            raise SystemExit(f"Token in {token_file} is too short (at least 16 characters)")
        self.profiles: dict[str, dict] = config.get("profiles") or DEFAULT_PROFILES
        self.routes: dict[str, list[str]] = (config.get("speaker_test") or {}).get("routes") or {}
        self.process: subprocess.Popen | None = None
        self.deployment_process: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.speaker_lock = threading.Lock()
        self.extraction_config = config.get("extraction") or {}
        self.extractor = Extractor(self.extraction_config, self.project.config["language"])
        self.extraction_lock = threading.Lock()
        self.resources = Resources(self.project.root, config.get("pause_services") or [])
        self.resources.release()
        self.system = SystemCheck(config.get("resource_check"))
        self.resource_warning = ""
        self.watcher: threading.Thread | None = None
        deployment_path = self.project.path("deployment.json")
        if deployment_path.is_file():
            deployment = json.loads(deployment_path.read_text())
            pid = deployment.get("pid")
            if deployment.get("state") == "running" and type(pid) is int:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    from .deployment import record

                    record(self.project, "failed", reason="Interrupted rollout; retry deployment")
        state = read(self.project)
        if state.get("state") in ("running", "starting"):
            State(self.project).update(state="failed", ended_at=now(), last_error="Interrupted: the service restarted")

        from .scheduler import Scheduler

        self.automatic = Scheduler(self, config.get("automatic_training") or {})

    # -- Runs -------------------------------------------------------------------

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, profile_id: str) -> tuple[int, dict]:
        profile = self.profiles.get(profile_id)
        if profile is None:
            return 400, {"error": "unknown profile", "profiles": sorted(self.profiles)}
        with self.lock:
            deployment_path = self.project.path("deployment.json")
            deploying = deployment_path.is_file() and json.loads(deployment_path.read_text()).get("state") == "running"
            if (
                deploying
                or self.running()
                or (self.deployment_process is not None and self.deployment_process.poll() is None)
            ):
                return 409, {"error": "a run or firmware rollout is active"}
            self.project = Project.load(self.project.root)
            command = [sys.executable, "-m", "wake_word_trainer", "-p", str(self.project.root)]
            if self.downloads:
                command += ["--downloads", str(self.downloads)]
            command += ["run", "--profile", profile_id]
            if profile.get("prepare_only"):
                command.append("--prepare-only")
            if "rounds" in profile:
                command += ["--rounds", str(int(profile["rounds"]))]
            if "steps" in profile:
                command += ["--steps", str(int(profile["steps"]))]
            problems = self.system.before_start()
            if problems:
                # Nothing started: the last run's state stays as it was.
                return 503, self._not_enough(problems, record=False)
            State(self.project).update(state="starting", profile=profile_id, started_at=now(), message="")
            log = self.project.path("service.log").open("a", encoding="utf-8")
            try:
                self.resources.release()
                self.resources.acquire()
                problems = self.system.after_pause()
                if problems:
                    self.resources.release()
                    return 503, self._not_enough(problems)
                self.process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=self.project.root
                )
            except Exception as err:
                self.resources.release()
                State(self.project).update(state="failed", ended_at=now(), last_error=str(err))
                return 503, {"error": "could not prepare training resources"}
            finally:
                log.close()
            self.resource_warning = ""
            self.watcher = threading.Thread(target=self._finished, args=(self.process,), daemon=True)
            self.watcher.start()
            threading.Thread(target=self._watch_resources, args=(self.process,), daemon=True).start()
        return 202, {"started": profile_id}

    def _not_enough(self, problems: list[str], record: bool = True) -> dict:
        if record:
            message = "Not enough free system resources: " + "; ".join(problems)
            State(self.project).update(state="failed", ended_at=now(), last_error=message)
        return {"error": "not enough free system resources: " + "; ".join(problems), "problems": problems}

    def _watch_resources(self, process: subprocess.Popen, interval: float = 30) -> None:
        """A run that has to swap barely moves: say so in the status instead of looking stuck."""
        while process.poll() is None:
            problems = self.system.while_running()
            warning = "; ".join(problems)
            if warning and warning != self.resource_warning:
                with self.project.path("service.log").open("a", encoding="utf-8") as log:
                    log.write(f"Resource warning: {warning}\n")
            self.resource_warning = warning
            try:
                process.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                pass

    def _finished(self, process: subprocess.Popen) -> None:
        process.wait()
        with self.lock:
            if self.process is not process:
                return
            try:
                self.resources.release()
            except Exception as err:
                State(self.project).update(state="failed", ended_at=now(), last_error=f"Resource restore failed: {err}")
                return

            # Restore GPU services before compiling firmware. Reserve the rollout under
            # the same lock as start(), then wait without holding up HTTP requests.
            deployment = self.project.path("deployment.json")
            ready = deployment.is_file() and json.loads(deployment.read_text()).get("state") == "ready"
            trained = read(self.project).get("best_model_available")
            if process.returncode == 0 and trained and ready and self.project.config["deployment"]["enabled"]:
                with self.project.path("deployment.log").open("a") as log:
                    self.deployment_process = subprocess.Popen(
                        [sys.executable, "-m", "wake_word_trainer.deployment", str(self.project.root)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
        if self.deployment_process is not None:
            self.deployment_process.wait()

    def stop(self) -> tuple[int, dict]:
        with self.lock:
            if not self.running():
                self.resources.release()
                return 200, {"stopped": False}
            assert self.process is not None
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=10)
            self.resources.release()
        state = read(self.project)
        if state.get("state") in ("running", "starting"):
            State(self.project).update(state="stopped", ended_at=now(), message="Stopped")
        return 200, {"stopped": True}

    def status(self) -> dict:
        state = read(self.project)
        if state.get("state") in ("running", "starting") and not self.running():
            state["state"] = "failed"
            state["last_error"] = state.get("last_error") or "The run ended unexpectedly; see service.log"
        deployment_path = self.project.path("deployment.json")
        if deployment_path.is_file():
            state["deployment"] = json.loads(deployment_path.read_text())
        state.update(
            automatic_training=dict(self.automatic.status),
            workstation_online=True,
            wake_word=self.project.config["wake_word"],
            slug=self.project.slug,
            gpu=gpu(),
            resources={**self.system.snapshot(), "warning": self.resource_warning if self.running() else ""},
            profiles={key: value.get("label", key) for key, value in self.profiles.items()},
            speaker_routes=sorted(self.routes),
            extraction={
                "enabled": bool(self.extraction_config.get("enabled", False)),
                "available": importlib.util.find_spec("faster_whisper") is not None,
            },
        )
        return state

    # -- Speaker test -------------------------------------------------------------

    def speaker_test(self, route: str) -> tuple[int, dict]:
        command = self.routes.get(route)
        if command is None:
            return 400, {"error": "unknown route", "routes": sorted(self.routes)}
        clips = audio.wavs(self.project.recordings / "eval") or audio.wavs(self.project.recordings / "train")
        if not clips:
            return 404, {"error": "no recordings; run fetch first"}
        clip = random.choice(clips)
        if not self.speaker_lock.acquire(blocking=False):
            return 409, {"error": "already playing"}
        try:
            result = subprocess.run(
                [part.replace("{file}", str(clip)) for part in command], capture_output=True, timeout=30, check=False
            )
        finally:
            self.speaker_lock.release()
        if result.returncode != 0:
            return 500, {"error": "playback failed", "detail": result.stderr.decode(errors="replace")[-300:]}
        return 200, {"route": route, "clip": str(clip.relative_to(self.project.root))}

    def extract(self, body: bytes, phrases: list[str]) -> tuple[int, dict]:
        if not self.extraction_config.get("enabled", False):
            return 503, {"error": "extraction disabled; enable extraction in service.yaml"}
        if (
            not isinstance(phrases, list)
            or not 1 <= len(phrases) <= 32
            or any(not isinstance(p, str) or not 1 <= len(p) <= 120 for p in phrases)
        ):
            return 400, {"error": "invalid phrases"}
        if not self.extraction_lock.acquire(blocking=False):
            return 409, {"error": "extraction busy; retry later"}
        try:
            return 200, self.extractor.extract(body, phrases)
        except ValueError:
            return 400, {"error": "invalid mono PCM16 WAV (0.5-120 seconds, 16/48 kHz)"}
        except ImportError:
            return 503, {"error": "install wake-word-trainer[segment]"}
        except Exception:
            return 503, {"error": "local speech recognition failed; check extraction model configuration"}
        finally:
            self.extraction_lock.release()

    # -- Files ----------------------------------------------------------------------

    def model_file(self, name: str) -> Path | None:
        if not MODEL_FILE.match(name) or not name.startswith(self.project.slug + "."):
            return None
        path = self.project.export_dir / name
        return path if path.is_file() else None


def gpu() -> dict:
    """GPU load, best effort: AMD (rocm-smi) or NVIDIA (nvidia-smi)."""
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        try:
            use, used, total = (float(value) for value in result.stdout.splitlines()[0].split(","))
            return {"use_percent": use, "memory_percent": round(100 * used / total, 1)}
        except (ValueError, IndexError, ZeroDivisionError):
            return {}
    if shutil.which("rocm-smi"):
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmemuse", "--json"], capture_output=True, text=True, timeout=5, check=False
        )
        try:
            card = next(iter(json.loads(result.stdout).values()))
            use = float(str(card.get("GPU use (%)", 0)).strip("%"))
            memory = float(str(card.get("GPU Memory Allocated (VRAM%)", card.get("GPU memory use (%)", 0))).strip("%"))
            return {"use_percent": use, "memory_percent": memory}
        except (ValueError, StopIteration, AttributeError, json.JSONDecodeError):
            return {}
    return {}


def handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        server_version = "wake-word-trainer"

        def log_message(self, *args):
            pass

        def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            header = self.headers.get("Authorization", "")
            given = header[7:] if header.startswith("Bearer ") else header
            return hmac.compare_digest(given.encode(), service.token.encode())

        def _body(self) -> dict:
            length = min(int(self.headers.get("Content-Length") or 0), 4096)
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                return self._send(200, {"ok": True})
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            if path == "/v1/status":
                return self._send(200, service.status())
            if path == "/v1/report":
                report = service.project.path("report.json")
                return self._send(200, report.read_bytes()) if report.is_file() else self._send(404, {})
            if path.startswith("/v1/model/"):
                model = service.model_file(path.rsplit("/", 1)[-1])
                if model is None:
                    return self._send(404, {"error": "no exported model"})
                kind = "application/json" if model.suffix == ".json" else "application/octet-stream"
                return self._send(200, model.read_bytes(), kind)
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            if self.path == "/v1/extract":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.headers.get("X-Wakeword-Phrases", "[]")
                    if len(raw) > 4096 or not 44 <= length <= MAX_BYTES:
                        return self._send(413, {"error": "invalid request size"})
                    phrases = json.loads(raw)
                    self.connection.settimeout(30)
                    audio_body = self.rfile.read(length)
                    if len(audio_body) != length:
                        return self._send(400, {"error": "incomplete WAV"})
                except (ValueError, TimeoutError):
                    return self._send(400, {"error": "invalid extraction request"})
                return self._send(*service.extract(audio_body, phrases))
            body = self._body()
            if self.path == "/v1/start":
                return self._send(*service.start(str(body.get("profile", "recommended"))))
            if self.path == "/v1/stop":
                return self._send(*service.stop())
            if self.path == "/v1/speaker_test":
                return self._send(*service.speaker_test(str(body.get("route", ""))))
            return self._send(404, {"error": "not found"})

    return Handler


def make_server(config: dict[str, Any]) -> tuple[ThreadingHTTPServer, Service]:
    service = Service(config)
    server = ThreadingHTTPServer((config.get("bind", "0.0.0.0"), int(config.get("port", 10701))), handler(service))
    return server, service


def serve(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    server, service = make_server(config)
    service.automatic.start()

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    print(f"Serving on {server.server_address[0]}:{server.server_address[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.automatic.close()
        service.stop()
        server.server_close()
