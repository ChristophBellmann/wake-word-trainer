# Trainer service for Home Assistant

`wake-word-trainer serve` runs on the training computer and lets Home
Assistant (the [Wake Word Collector](https://github.com/ChristophBellmann/ha-wake-word-collector)
integration) start and stop trainings, show progress and results, take over
the finished model and run a loudspeaker test against a satellite.

## Configuration

Copy [`deploy/service.example.yaml`](https://github.com/ChristophBellmann/wake-word-trainer/blob/main/deploy/service.example.yaml)
to `~/.config/wake-word-trainer/service.yaml` and create a token:

```bash
(umask 077; openssl rand -hex 24 > ~/.config/wake-word-trainer/service.token)
```

```yaml
project: ~/wakeword/hey_jarvis              # project folder with wakeword.yaml
downloads: ~/wakeword/downloads             # optional: shared downloads
token_file: ~/.config/wake-word-trainer/service.token   # >= 16 characters
bind: 0.0.0.0
port: 10701
profiles:                                   # what Home Assistant can start
  quick: {label: Quick, rounds: 0, steps: 10000}
  recommended: {label: Recommended, rounds: 2}
  thorough: {label: Thorough, rounds: 4, steps: 40000}
  validate: {label: Check data only, prepare_only: true}
speaker_test:                               # optional: play held-out recordings
  routes:
    speakers: [paplay, "{file}"]
    usb: [aplay, -D, "plughw:2,0", "{file}"]
```

Network callers can only choose among configured **profiles** and **routes**:
no commands, paths or settings reach the service from Home Assistant.

## Run it as a systemd user service

```bash
mkdir -p ~/.config/systemd/user
cp deploy/wake-word-trainer.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now wake-word-trainer
loginctl enable-linger $USER   # keep it running without a login session
```

The unit starts `%h/wake-word-trainer/.venv/bin/wake-word-trainer serve
--config %h/.config/wake-word-trainer/service.yaml`; adapt the path to your
virtual environment.

## Connect Home Assistant

In the Wake Word Collector's options enter `http://<training computer>:10701`
and the service token. The collector then shows the training state, offers
the profiles, and can take over the exported model and roll it out.

When the training computer is switched off the collector shows the trainer
as unreachable; that is normal and nothing is lost.

## Loudspeaker test

With `speaker_test.routes` the service plays held-out recordings through a
loudspeaker next to a satellite while Home Assistant counts the detections:
a measurement with real sound, room and microphone.

More: [sharing the GPU and checking resources](resources.md),
[clip extraction](extraction.md), [automatic deployment](deployment.md),
[idle-night training](automatic-training.md), [HTTP API](api.md).
