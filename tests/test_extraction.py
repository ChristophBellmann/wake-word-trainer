"""Word alignment, independent recognition, and bounded authenticated HTTP."""

import io
import json
import threading
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from wake_word_trainer.extraction import MAX_BYTES, Extractor, find_phrases, validate_audio
from wake_word_trainer.service import Service, handler


def wav(seconds=30):
    out = io.BytesIO()
    with wave.open(out, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\0\0" * int(seconds * 16000))
    return out.getvalue()


def word(text, start, end, probability=0.9):
    return dict(word=text, start=start, end=end, probability=probability)


def test_complete_phrases_variants_and_other_speech():
    words = [
        word("Hey,", 1, 1.3),
        word("Növa!", 1.35, 1.9),
        word("Licht", 2, 2.5),
        word("Hei", 5, 5.4),
        word("Nova", 5.5, 6),
        word("Hey", 9, 9.3),
        word("Nora", 9.4, 10),
    ]
    result = find_phrases(words, ["hey nova", "hei nova"], 30, {})
    assert len(result) == 2
    assert result[0]["start_ms"] == 800 and result[0]["end_ms"] == 1950
    assert result[1]["phrase"] == "hei nova"


def test_quiet_low_confidence_incomplete_and_disconnected_words():
    assert find_phrases([], ["hey nova"], 30, {}) == []
    assert find_phrases([word("Hey", 1, 1.5)], ["hey nova"], 30, {}) == []
    assert find_phrases([word("Hey", 1, 1.5), word("Nova", 4, 4.5)], ["hey nova"], 30, {}) == []
    assert find_phrases([word("Hey", 1, 1.5, 0.1), word("Nova", 1.6, 2, 0.1)], ["hey nova"], 30, {}) == []
    assert find_phrases([word("Hey", float("nan"), 1.5), word("Nova", 1.6, 2)], ["hey nova"], 30, {}) == []


def test_adjacent_repetitions_do_not_overlap():
    result = find_phrases(
        [word("Hey", 0, 0.3), word("Nova", 0.4, 1), word("Hey", 1.1, 1.4), word("Nova", 1.5, 2)], ["hey nova"], 2, {}
    )
    assert len(result) == 2 and result[0]["end_ms"] <= result[1]["start_ms"]
    assert result[0]["start_ms"] == 0 and result[-1]["end_ms"] == 2000


def test_extractor_uses_separate_speech_regions_and_word_times_without_target_prompt():
    calls = []

    class Model:
        def transcribe(self, path, **kwargs):
            calls.append((path, kwargs))
            words = [SimpleNamespace(**word("Hey", 1, 1.4)), SimpleNamespace(**word("Nova", 1.5, 2))]
            return iter([SimpleNamespace(words=words, no_speech_prob=0.1, avg_logprob=-0.2)]), None

    extractor = Extractor({}, "de")
    extractor.model = Model()
    extractor._speech_regions = lambda samples: [{"start": 0, "end": 3 * 16000}, {"start": 5 * 16000, "end": 8 * 16000}]
    result = extractor.extract(wav(), ["hey nova"])
    assert len(result["segments"]) == 2
    assert result["segments"][1]["start_ms"] == 5800
    assert len(calls) == 2 and calls[0][0].shape == (48000,)
    assert calls[0][1]["vad_filter"] is False
    assert calls[0][1]["without_timestamps"] is True
    assert calls[0][1]["word_timestamps"] and calls[0][1]["language"] == "de"
    assert "initial_prompt" not in calls[0][1]


@pytest.mark.parametrize("body", [b"not audio", wav(0.1), wav(121), wav()[:-20]])
def test_invalid_audio(body):
    with pytest.raises(ValueError):
        validate_audio(body)


def test_http_auth_body_limits_and_service_extraction():
    service = Service.__new__(Service)
    service.token = "secret-for-test-only"
    service.extraction_config = {"enabled": True}
    service.extraction_lock = threading.Lock()
    service.extractor = SimpleNamespace(
        extract=lambda body, phrases: {"segments": [], "duration_ms": int(validate_audio(body) * 1000)}
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1/extract"

    def post(body, auth=True, **headers):
        h = {"X-Wakeword-Phrases": json.dumps(["hey nova"]), **headers}
        if auth:
            h["Authorization"] = "Bearer " + service.token
        request = urllib.request.Request(url, body, h)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as err:
            return err.code, json.load(err)

    try:
        assert post(wav(0.5), auth=False)[0] == 401
        assert post(wav())[1]["duration_ms"] == 30000
        assert post(wav(0.5), **{"X-Wakeword-Phrases": "broken"})[0] == 400
        assert post(wav(0.5), **{"X-Wakeword-Phrases": '"not a list"'})[0] == 400
        assert post(b"small", **{"Content-Length": str(MAX_BYTES + 1)})[0] == 413
        service.extraction_lock.acquire()
        assert post(wav())[0] == 409
        service.extraction_lock.release()
        service.extraction_config["enabled"] = False
        assert post(wav())[0] == 503
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_merged_transcription_uses_configured_variant_not_fuzzy_matching():
    words = [word("Haimomo", 1, 1.8)]
    assert find_phrases(words, ["hey momo"], 30, {}) == []
    result = find_phrases(words, ["hey momo", "hai momo"], 30, {})
    assert len(result) == 1 and result[0]["phrase"] == "hai momo"
