# Data licenses and privacy

## Data licenses

The trainer itself is Apache-2.0. The downloaded datasets (AudioSet, Free
Music Archive, MIT impulse responses, microWakeWord's negative sets,
Multilingual LibriSpeech) have different licenses and usage restrictions. As
microWakeWord states for the same data: models trained with them are for
personal, non-commercial use.

Piper voices have their own licenses; check the model card of every voice you
download.

## Privacy

- Your recordings stay in the project folder on your computer.
- The collector export is protected by a token; the trainer fetches, Home
  Assistant never pushes recordings anywhere else.
- Clip extraction processes audio in memory on the training computer and
  does not store it; nothing is sent to a cloud.
- Do not publish projects or models trained on other people's voices without
  their consent.
