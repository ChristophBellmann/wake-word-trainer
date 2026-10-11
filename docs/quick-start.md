# Quick start

With the [Wake Word Collector](https://github.com/ChristophBellmann/ha-wake-word-collector)
in Home Assistant. Without it, put WAV files of the wake word into a folder
and list it under `recordings.folders` instead of the collector settings.

## 1. Token

In Home Assistant, *Wake Word Collector → Configure* shows the export token.
Save it:

```bash
mkdir -p ~/.config/wake-word-trainer
(umask 077; echo "<token>" > ~/.config/wake-word-trainer/hey_jarvis.token)
```

## 2. Project, voices and data

```bash
wake-word-trainer -p hey_jarvis init "Hey Jarvis" --language en \
  --collector-url http://homeassistant.local:8123 \
  --token-file ~/.config/wake-word-trainer/hey_jarvis.token
wake-word-trainer -p hey_jarvis download --voice en_US-lessac-medium --voice en_US-amy-medium --data
```

`--data` downloads the shared background audio, impulse responses and
negative features once (about 15 GB).

## 3. Phrases

Edit `hey_jarvis/wakeword.yaml`:

```yaml
tts:
  voices: [voices/en_US-lessac-medium.onnx, voices/en_US-amy-medium.onnx]
  phrases: [hey jarvis, hey jarviss]
  negative_phrases: [hey jarvo, hey travis, hey service, jarvis, hey]
```

`phrases` is what is synthesised; phonetic spellings help.
`negative_phrases` are look-alikes the model must ignore.

## 4. Train, evaluate, export

```bash
wake-word-trainer -p hey_jarvis run
```

The end of the output looks like this (numbers made up):

```text
cutoff 0.906: recognizes 93.8% of 48 own recordings, 0.31 false activations per hour (8.3 h background)
Not recognized (3): listen to them; mislabelled clips belong rejected in the collector
```

## 5. Use it

Put `export/hey_jarvis.json` and `export/hey_jarvis.tflite` where ESPHome
finds them (beside the device YAML, or on a web server):

```yaml
micro_wake_word:
  models:
    - model: hey_jarvis.json
```

## Again

Record more in the collector, then `run` again: only new clips are fetched,
the synthetic speech is kept, training starts over with the new data.

German example: `--language de`, voices `de_DE-thorsten-medium`,
`de_DE-kerstin-low`, and German speech as negatives with
`download --speech mls_de` (see [better models](better-models.md)).
