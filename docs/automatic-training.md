# Training during idle nights

The service can automatically train new accepted positives and negatives:

```yaml
automatic_training:
  enabled: true
  start: "01:00"                 # workstation local time
  end: "06:00"
  idle_seconds: 1200             # no desktop input for 20 minutes
  idle_command: [/path/to/venv/bin/python, -m, wake_word_trainer.idle]
  min_new_samples: 5
  min_interval_hours: 20         # at most one automatic attempt per night
  profile: recommended
  max_gpu_use_percent: 5         # unknown GPU activity blocks automation
  max_load_per_cpu: 0.5
  check_seconds: 30
```

`wake_word_trainer.idle` measures the X11 screensaver's input inactivity using
libX11/libXss; the service needs its desktop's `DISPLAY`, `XAUTHORITY` and
`XDG_SESSION_TYPE=x11`. On other desktops configure a command that prints
idle **seconds**, not a constant. An unavailable or invalid probe blocks the
automatic run. GPU use and CPU load are checked before starting; use
`max_gpu_use_percent: null` for a CPU-only workstation.

The first check establishes a baseline of collector sample hashes. New data
arriving during training stays pending for the next run. A failed or interrupted
run does not consume examples. Cooldown prevents repeated failed attempts in
one night. When desktop input resumes or the night window closes, only the
automatically started run stops; GPU services are restored. Manual runs are
never interrupted by this policy. Existing firmware deployment blocks a new
training. `/v1/status.automatic_training` explains why the scheduler is waiting.

In the Collector enable **Learn from activations without input** and install
its matching ESPHome package. A confirmed empty STT result labels the buffered
activation audio as a negative; technical failures remain for review. The
normal held-out comparison still gates each automatic deployment. Training on
new negatives cannot guarantee an improvement: an equal or worse model is
kept out of the devices.
