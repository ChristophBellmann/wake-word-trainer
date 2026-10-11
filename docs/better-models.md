# Better models: more data, fewer false activations

## Your recordings decide

Synthetic voices are cheap, but a model that is good on them is not
necessarily good on you. 50 or more own recordings from the satellites, from
several people and distances, make a real difference. The
[Wake Word Collector](https://github.com/ChristophBellmann/ha-wake-word-collector)
records them with the satellites' own microphones.

## Your own negatives

Recordings **without** the wake word from your home — TV, conversations,
words that sound alike — are the strongest lever against false activations:

- `recordings.negative_folders`, or clips marked *not the wake word* in the
  collector;
- false alarms reported on the satellites (*"false alarm"* in the collector);
- with the collector's *learn from activations without input*, activations
  after which nobody said anything.

A held-out share also limits the cutoff (`max_own_negative_share`), and the
report lists the ones that still trigger.

## Speech in your language

The microWakeWord negative sets are mostly English.

```bash
wake-word-trainer -p hey_jarvis download --speech mls_de
```

fetches German audio books (Multilingual LibriSpeech; also `mls_fr`,
`mls_nl`, `mls_es`, `mls_it`, `mls_pt`, `mls_pl`). Add the preset to
`negatives.speech_presets`.

## Mining rounds

`training.mining_rounds: 2` or `run --rounds 2` lets the model find its own
weak spots in the negative training data: after each round it collects what
it wrongly reacts to and trains on, with extra weight.

## Your rooms

`augmentation.background_folders` with recordings of your own living room and
`augmentation.rir_folders` with impulse responses make the augmentation sound
like home.

## Good to know

- **One model per wake word.** Two phrases in one model compete; train them
  separately.
- **Short words are hard.** Three or four syllables ("Hey Jarvis") work much
  better than one ("Jarvis").
- **`training.seed`** makes augmentation, sampling and initialisation
  repeatable for comparisons.
