# Replacing a model you already use

Measure before you switch:

```bash
wake-word-trainer -p hey_jarvis compare --current --model /path/to/old_model.tflite
```

With the old manifest (`old_model.json`) beside the `.tflite`, both models
are measured on your held-out recordings:

- each at its **own** cutoff from its manifest,
- and each at the cutoff **your budget** allows.

Switch only when the new one recognises at least as much with no more false
activations. `--model` can be given several times, for example to compare the
official `hey_jarvis` model with your own.

[Automatic deployment](deployment.md) applies the same comparison before it
touches a satellite.
