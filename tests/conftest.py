"""Shared helpers: synthetic audio and a fake Home Assistant / Hugging Face server."""

from __future__ import annotations

import http.server
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from wake_word_trainer import audio

RATE = 16000


def chirp(seconds: float = 0.8, seed: int = 0) -> np.ndarray:
    """A recognizable 'word': rising tone with a little variation per seed."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    start, end = 400 + rng.uniform(-40, 40), 1800 + rng.uniform(-100, 100)
    phase = 2 * np.pi * (start * t + (end - start) * t**2 / (2 * seconds))
    return (0.4 * np.sin(phase) * np.hanning(len(t))).astype(np.float32)


def noise(seconds: float, seed: int = 0, level: float = 0.05) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(seconds * RATE)) * level).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray) -> bytes:
    audio.write(path, samples)
    return path.read_bytes()


class FakeServer:
    """Serves {path: bytes | callable(headers) -> (status, bytes)}."""

    def __init__(self, routes: dict[str, bytes | Callable]):
        self.routes = routes
        self.requests: list[str] = []
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                server.requests.append(self.path)
                route = server.routes.get(self.path.split("?")[0])
                status, body = (
                    (404, b"") if route is None else (route(self.headers) if callable(route) else (200, route))
                )
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    servers = []

    def start(routes):
        servers.append(FakeServer(routes))
        return servers[-1]

    yield start
    for item in servers:
        item.close()
