"""Generate varied WAVs with Piper's ONNX API, without PyTorch or piper_train.

Run in a subprocess so ONNX voices release their memory before training.
"""

from __future__ import annotations

import argparse
import itertools
import wave
from pathlib import Path


def generate(text: str, models: list[str], count: int, target: Path, scales: list[float]) -> None:
    from piper import PiperVoice, SynthesisConfig

    voices = [PiperVoice.load(model, use_cuda=False) for model in models]
    settings = [
        (voice, speaker, scale) for voice in voices for speaker in range(voice.config.num_speakers) for scale in scales
    ]
    target.mkdir(parents=True, exist_ok=True)
    for index, (voice, speaker, scale) in enumerate(itertools.islice(itertools.cycle(settings), count)):
        path = target / f"{index:06d}.wav"
        if path.is_file():
            continue
        # Interrupted synthesis must not leave a WAV that the next run counts as complete.
        part = path.with_suffix(".wav.part")
        with wave.open(str(part), "wb") as output:
            voice.synthesize_wav(text, output, syn_config=SynthesisConfig(speaker_id=speaker, length_scale=scale))
        part.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length-scales", type=float, nargs="+", default=[1.0])
    args = parser.parse_args()
    generate(args.text, args.model, args.max_samples, args.output_dir, args.length_scales)


if __name__ == "__main__":
    main()
