# Entwicklung

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
ruff check .
pytest -q -m 'not slow'
```

Die schnellen Tests prüfen Datenrollen, SHA-Zuordnung, Split-Isolation,
Konfiguration, Dienste, Ressourcen und Prüfberichte mit künstlichem Audio.
Langsame Tests benötigen eine passende TensorFlow-/microWakeWord-Umgebung;
sie trainieren und vergleichen kleine Modelle. GPU-Geräteerkennung ersetzt
keinen tatsächlichen GPU-Rechenschritt.

ROCm und NVIDIA/CUDA verwenden denselben Trainingscode mit getrennten
passenden Framework-Wheels. Backend-Auswahl, Loader-Pfade und Fehler bei
fehlender GPU werden gezielt geprüft. Einen auf AMD ausgeführten Benchmark
nicht als NVIDIA-Hardwaretest ausgeben.

Bei fachlichen Änderungen die zugehörigen Seiten unter `docs/` und
`SUMMARY.md` aktualisieren. `.gitbook.yaml` und `gitbook-docs.yaml` zeigen
beide auf diesen Ordner. Öffentliche Texte bleiben allgemein, auf Deutsch
und ohne persönliche Installationshistorien oder Sprachdaten.
