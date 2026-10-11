# Clip extraction from long recordings

Install the optional `segment` extra (`pip install ".[segment]"`, using your
existing constraints on ROCm) and enable `extraction` in the service YAML:

```yaml
extraction:
  enabled: true
  model: small
  cpu_threads: 4
  padding_ms: 200
  min_probability: 0.35
```

The Collector sends a manual recording to authenticated `POST /v1/extract`
(WAV body, `X-Wakeword-Phrases` JSON list of its configured phrase and variants).
Local faster-whisper CPU/int8 recognition produces word timestamps. Use a current
model conversion with `alignment_heads` metadata (the default `small` download
provides it); older Wyoming model folders may lack word alignment support. The
service returns complete matching phrases with bounded padding; it does not
write recordings into the training project. Audio is processed in memory and is not stored on the workstation. Models may be downloaded once; audio is never sent to a cloud.
The project's `language` selects the recognition language. The model is
independent of the wake-word model being trained, so missed wake-word
activations do not exclude examples. Extraction runs serially on the CPU
without pausing GPU services or changing the active training run.

Requests are bounded to 4 MiB and 0.5–120 seconds of mono PCM16 WAV (16/48 kHz).
`/v1/status` advertises `extraction.enabled` and `extraction.available`.
Recognition is imperfect: difficult pronunciations may need additional
Collector variants or manual editing. Clips must be listened to and accepted
in the Collector before training. Speech regions are recognized separately so repeated identical phrases are
not collapsed by transcription. `silence_ms` (default 500) controls separation.
A second between repetitions helps cutting.

The collector shows extracted clips as *to check*; the original recording
stays in Home Assistant.
