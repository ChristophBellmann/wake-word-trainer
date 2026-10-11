# Automatic firmware deployment

Training completion alone does not update a satellite. To opt into automatic
OTA, add this to the private project's `wakeword.yaml`:

```yaml
deployment:
  enabled: true
  reference_model: deployed/model.tflite
  command: [python3, /path/to/model_update.py, --config, /path/to/wake-word-model.yaml,
            --expected-sha256, "{sha256}", --wait, "120", --build]
```

First copy the model currently running on the devices **and its adjacent
manifest** to `deployed/model.tflite` and `deployed/model.json`. Device paths,
network access, credentials and flash commands stay local; the HTTP client
cannot set them. The command may run over SSH on the ESPHome host. It must
verify every requested device and return nonzero on any failed deployment.
Use the Collector's public `model_update.py` with `require_parity: true` and
`{hash}` in its model filename.

The pipeline evaluates the new export and deployed reference on the same
held-out own recordings and background audio. It compares both at their
manifest thresholds. Automatic deployment requires no regression in recall or
own negative triggers, an improvement in at least one of them, and compliance
with the configured false activation budgets. Synthetic-only evaluation cannot
approve automatic OTA. The report includes a comparison bound to both model
SHA256 values before Home Assistant receives completion.

After training, the service restores the paused GPU services, then runs the
configured deployment command. `/v1/status` includes a separate `deployment`
state; failed firmware updates keep the completed training and exported model.
The deployed reference advances only when **all** devices have succeeded.
Retry with `wake-word-trainer -p <project> deploy`; to compare an already
completed export first, use `deploy --compare`. Local details are in
`deployment.json`, `parity.json` and `deployment.log`. The Collector tool keeps
per-device progress and skips successful devices on retries.
