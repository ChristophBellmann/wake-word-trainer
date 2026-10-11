# Wake Word Trainer

Train a wake word model for ESPHome voice satellites (`micro_wake_word`) from
**your own recordings** plus synthetic speech, and get an honest measurement
of how well it works on your voice before you flash it.

It is the second half of the
[Wake Word Collector](https://github.com/ChristophBellmann/ha-wake-word-collector):

```text
satellite ──► Wake Word Collector (Home Assistant) ──► Wake Word Trainer ──► satellite
  say it        record, sort, review, export             fetch, train,         micro_wake_word
                                                         evaluate, export
```

It also works without the collector, from folders of WAV files or synthetic
speech alone. Training uses [microWakeWord](https://github.com/kahrendt/microWakeWord)
and [Piper](https://github.com/OHF-Voice/piper1-gpl) voices through Piper's
ONNX API.

## The way through

1. [Install](installation.md) the trainer on a computer with a GPU.
2. Follow the [quick start](quick-start.md): create a project, download
   voices and data, `run`.
3. Read the [evaluation](evaluation.md): how many of your own held-out
   recordings it recognises, how many false activations per hour.
4. Put the exported model on your satellites — by hand, or let Home Assistant
   do it through the [trainer service](service.md) and
   [automatic deployment](deployment.md).
5. Record more, fetch again, train again. With
   [idle-night training](automatic-training.md) this happens by itself.

## What makes it different

- **Your recordings decide.** Evaluation uses only your own held-out
  recordings, split by content hash, so the test set never leaks into
  training.
- **An honest cutoff.** The model runs exactly like on the satellite, over
  all 256 cutoffs; the most sensitive one within your false-activation budget
  is chosen.
- **Hard negatives from your home.** Clips marked *not the wake word* in the
  collector and mined false activations become training data.
- **No regression on the devices.** Automatic deployment only replaces a
  model that is at least as good on the same data.
- **Runs beside other GPU work.** Pauses your LLM or TTS services while
  training and refuses to start without free memory.

## Project

- Source and issues: [github.com/ChristophBellmann/wake-word-trainer](https://github.com/ChristophBellmann/wake-word-trainer)
- Releases: [GitHub releases](https://github.com/ChristophBellmann/wake-word-trainer/releases)
- License: Apache-2.0 (downloaded datasets have their own, see
  [licenses](licenses-and-privacy.md))

Deutsch: [Überblick auf Deutsch](deutsch.md).
