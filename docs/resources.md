# Sharing the GPU and checking resources

## Pause other GPU services

When an LLM or GPU TTS runs on the training computer as well, list their
systemd user services in `service.yaml`:

```yaml
pause_services:
  - llama-server.service
  - orpheus-llm.service
  - wyoming-orpheus.service
```

Before a run the trainer stops the services that were **active**. After the
run, an error, *Stop training* or a regular shutdown of the trainer service it
starts exactly these services again; services that were inactive stay
inactive. An interrupted run is recovered at the next start of the trainer;
the state lives in the project as `paused_services.json`. The names are
examples; without this setting no other service is touched. This applies to
`serve`; direct CLI runs do not manage other services.

## Free resources before and during a run

Paused services free VRAM, but not memory another process holds. A training
that has to swap hardly progresses. `serve` therefore checks before every
start (from Home Assistant, through the API and for the nightly training) and
refuses with HTTP 503 and the reason:

```yaml
resource_check:                 # all values optional
  min_available_memory_gb: 6    # MemAvailable before and during the run
  max_load_per_cpu: 0.75        # load per CPU before the start
  min_free_vram_gb: 8           # after pausing services; not checked without it
  wait_seconds: 10              # time until freed VRAM becomes visible
```

- Refused starts do not change the state of the last run.
- If the VRAM check fails after pausing, the services are started again.
- During a run memory is checked every 30 seconds. `/v1/status` shows the
  values and a warning under `resources`, also written to `service.log`. The
  warning does not stop the run.
- `enabled: false` switches the check off.
