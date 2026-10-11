# Evaluation and the cutoff

## What is measured

`evaluate` runs the exported model **exactly like the satellite** (streaming,
int8, sliding window of `evaluation.sliding_window_size` frames):

- **Recall** on your own held-out recordings: which share it recognises.
- **False activations per hour** on hours of background audio (speech,
  music, dinner-party noise).
- **Own negatives**: how many of your held-out recordings without the wake
  word trigger it.

for all 256 possible cutoffs.

## Choosing the cutoff

The most sensitive cutoff is chosen that stays within

- `evaluation.max_false_accepts_per_hour` (default 0.5),
- `evaluation.max_own_negative_share` (default 5 % of your held-out
  negatives),

and not below `evaluation.min_probability_cutoff` (0.5). The cutoff is
written into the manifest `<slug>.json` as `probability_cutoff`.

**The cutoff is a trade-off.** A lower false-activation budget gives a
higher cutoff and fewer accidental activations, but more misses.

## The report

```text
cutoff 0.906: recognizes 93.8% of 48 own recordings, 0.31 false activations per hour (8.3 h background)
Not recognized (3): listen to them; mislabelled clips belong rejected in the collector
```

`report.json` holds the full curve, the misses and the own negatives that
still trigger. Through the [service](service.md) Home Assistant shows it.

## Listen to the misses

The report lists the recordings the model does not recognise. Often they are
mislabelled: something else was said, or it was cut off. Reject them in the
collector and fetch again.

## Synthetic only

Without own recordings the evaluation falls back to synthetic speech and the
report says so. Such numbers say little about your voice, and they never
approve an [automatic deployment](deployment.md).
