"""The project folder: one wake word, its settings (wakeword.yaml) and all data.

Layout (everything below the project folder, nothing elsewhere):

    wakeword.yaml
    recordings/train/<device>/*.wav    your own voice, used for training
    recordings/eval/<device>/*.wav     your own voice, held out for evaluation
    tts/positive/*.wav                 synthetic wake word
    tts/negative/*.wav                 synthetic phrases that sound similar
    downloads/                         room impulse responses, background audio, negative features
    features/                          spectrograms for training
    model/                             microWakeWord training output
    export/<slug>.tflite, <slug>.json  the model for ESPHome
    report.json                        evaluation of the last training
"""

from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = "wakeword.yaml"

DEFAULTS: dict[str, Any] = {
    "wake_word": "",
    "slug": "",
    "language": "en",
    # Own recordings: the Wake Word Collector export and/or local folders of 16 kHz WAVs.
    "collector": {"url": "", "token_file": ""},
    "recordings": {
        "folders": [],
        # Difficult but correct examples (far away, fast, unusual): trained with extra weight.
        "hard_folders": [],
        # Your own recordings WITHOUT the wake word, ideally real false activations.
        "negative_folders": [],
        "eval_share": 0.2,
    },
    # More speech the model must ignore, e.g. in your language (16 kHz WAV folders or
    # downloaded presets, see `download --speech`).
    "negatives": {"speech_folders": [], "speech_presets": [], "speech_clips": 6000},
    "tts": {
        # Piper voices (.onnx with .onnx.json beside it), relative to the project or absolute.
        "voices": [],
        # What the voices say. Phonetic spellings often sound better than the real one.
        "phrases": [],
        "samples": 2000,
        # Similar sounding phrases the model must ignore.
        "negative_phrases": [],
        "negative_samples": 1000,
        "length_scales": [0.8, 0.9, 1.0, 1.1, 1.2],
    },
    "augmentation": {
        "background_probability": 0.25,
        "background_min_snr_db": 6,
        "background_max_snr_db": 22,
        "rir_probability": 0.3,
        # Your own background recordings or impulse responses, in addition to the downloads.
        "background_folders": [],
        "rir_folders": [],
    },
    "training": {
        "seed": None,  # optional reproducible augmentation, sampling and model initialization
        "steps": 20000,
        "batch_size": 128,
        "learning_rate": 0.001,
        "positive_class_weight": 1.0,
        "negative_class_weight": 20.0,
        # How often each own training recording is augmented differently.
        "own_repeat": 8,
        # Sampling weights in the training batches.
        "weights": {
            "own": 3.0,
            "own_hard": 1.5,
            "own_negative": 4.0,
            # Recordings without the wake word from the Collector (satellite false
            # activations): own share, so a large local negative folder cannot dilute them.
            "satellite_negative": 4.0,
            "speech_extra": 5.0,
            "mined": 4.0,
            "tts": 2.0,
            "tts_negative": 3.0,
            "speech": 10.0,
            "dinner_party": 10.0,
            "no_speech": 5.0,
        },
        "clip_duration_ms": 1500,
        # Rounds of: train, find false activations in the background audio, train again with them.
        "mining_rounds": 0,
        "mining_samples": 20000,  # negative spectrograms checked per round
        "mining_threshold": 0.4,  # averaged score above which a negative counts as found
        "eval_step_interval": 500,
    },
    "evaluation": {
        # The cutoff is the lowest one with at most this many false activations per hour.
        "max_false_accepts_per_hour": 0.5,
        # Never more sensitive than this: the background set is only a sample of real life.
        "min_probability_cutoff": 0.5,
        # Share of your own held-out non-wake-word recordings that may still trigger.
        "max_own_negative_share": 0.05,
        "sliding_window_size": 5,
    },
    "deployment": {"enabled": False, "reference_model": "", "command": []},
    "export": {
        "author": "",
        "website": "",
        "tensor_arena_size": 30000,
        "minimum_esphome_version": "2024.7.0",
    },
}


class ProjectError(Exception):
    """A problem with the project folder or wakeword.yaml, with a readable message."""


def slugify(text: str) -> str:
    """'Hey Jarvis!' -> 'hey_jarvis'."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return "_".join(re.findall(r"[a-z0-9]+", text))


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class Project:
    root: Path
    config: dict[str, Any]

    # -- Loading --------------------------------------------------------------

    @classmethod
    def load(cls, root: str | Path) -> Project:
        root = Path(root).expanduser().resolve()
        path = root / CONFIG_NAME
        if not path.is_file():
            raise ProjectError(f"No {CONFIG_NAME} in {root}. Create one with: wake-word-trainer init")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as err:
            raise ProjectError(f"{path}: {err}") from err
        if not isinstance(data, dict):
            raise ProjectError(f"{path} must be a mapping")
        project = cls(root, _merge(DEFAULTS, data))
        project.validate()
        return project

    @classmethod
    def create(cls, root: str | Path, wake_word: str, language: str = "en", **settings: Any) -> Project:
        root = Path(root).expanduser().resolve()
        if (root / CONFIG_NAME).exists():
            raise ProjectError(f"{root / CONFIG_NAME} exists already")
        config = _merge(DEFAULTS, settings)
        config["wake_word"] = wake_word
        config["slug"] = slugify(wake_word)
        config["language"] = language
        if not config["tts"]["phrases"]:
            config["tts"]["phrases"] = [wake_word.lower()]
        project = cls(root, config)
        project.validate()
        root.mkdir(parents=True, exist_ok=True)
        (root / CONFIG_NAME).write_text(project.render_config(), encoding="utf-8")
        return project

    def render_config(self) -> str:
        """The settings that differ from the defaults, plus the essentials."""
        essentials = {key: self.config[key] for key in ("wake_word", "slug", "language")}
        changed = _diff(self.config, DEFAULTS)
        changed.update(essentials)
        ordered = {key: changed[key] for key in [*essentials, *[k for k in changed if k not in essentials]]}
        return (
            "# wake-word-trainer project; all settings and their defaults:\n"
            "# https://github.com/ChristophBellmann/wake-word-trainer#settings\n"
            + yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True)
        )

    def validate(self) -> None:
        config = self.config
        if not str(config["wake_word"]).strip():
            raise ProjectError("wake_word is empty")
        if not config["slug"]:
            config["slug"] = slugify(config["wake_word"])
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]{0,63}", config["slug"]):
            raise ProjectError("slug: lower case letters, digits and _ only")
        share = config["recordings"]["eval_share"]
        if not 0 < share < 1:
            raise ProjectError("recordings.eval_share must be between 0 and 1")
        if config["evaluation"]["max_false_accepts_per_hour"] <= 0:
            raise ProjectError("evaluation.max_false_accepts_per_hour must be positive")
        if not 0 <= float(config["evaluation"]["min_probability_cutoff"]) < 1:
            raise ProjectError("evaluation.min_probability_cutoff must be 0..1")
        if not 1 <= int(config["evaluation"]["sliding_window_size"]) <= 20:
            raise ProjectError("evaluation.sliding_window_size must be 1..20")
        if int(config["training"]["steps"]) < 1:
            raise ProjectError("training.steps must be positive")
        seed = config["training"].get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ProjectError("training.seed must be null or an integer in 0..4294967295")

        deployment = config["deployment"]
        if deployment["enabled"]:
            if not deployment["reference_model"]:
                raise ProjectError("deployment.reference_model is required")
            command = deployment["command"]
            if not isinstance(command, list) or not command or not all(isinstance(p, str) for p in command):
                raise ProjectError("deployment.command must be a nonempty list of command arguments")

    # -- Paths ----------------------------------------------------------------

    @property
    def slug(self) -> str:
        return self.config["slug"]

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def resolve(self, value: str) -> Path:
        """A path from wakeword.yaml: ~ expanded, relative to the project folder."""
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root / path

    @property
    def recordings(self) -> Path:
        return self.path("recordings")

    @property
    def downloads(self) -> Path:
        return self.path("downloads")

    @property
    def features(self) -> Path:
        return self.path("features")

    @property
    def model_dir(self) -> Path:
        return self.path("model")

    @property
    def export_dir(self) -> Path:
        return self.path("export")


def _diff(config: dict, defaults: dict) -> dict:
    result = {}
    for key, value in config.items():
        default = defaults.get(key)
        if isinstance(value, dict) and isinstance(default, dict):
            nested = _diff(value, default)
            if nested:
                result[key] = nested
        elif value != default:
            result[key] = value
    return result
