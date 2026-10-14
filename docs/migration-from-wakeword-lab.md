# Unterschiede und Übernahme aus dem Wakeword Lab

Verglichen wurde die Arbeitslinie `hey_momo` des alten Labs, einschließlich
`README.md`, `EXPERIENCES.md`, `docs/wakeword_flow.md`,
`docs/training_dataset.md`, `scripts/README.md`, `scripts/train_from_config.sh`,
`scripts/setup/setup_sync_datasets.py` und
`scripts/setup/setup_fetch_notebook_backgrounds.py`.

Das Lab enthält sowohl die funktionierende Pipeline als auch verworfene
Versuche. Deshalb werden Datenrollen und bewährte Funktionen übernommen;
experimentelle Einstellungen werden ausdrücklich gewählt. Die alten
Privatdaten, Konfigurationen, Transkripte und Modellgewichte gehören weiterhin
in private Projektordner.

## Funktionsvergleich

| Bereich | Altes Lab | Trainer vor der Übernahme | Übernahme / verbleibender Unterschied |
| --- | --- | --- | --- |
| Aufbau | Shell-Orchestrator, viele einzelne Skripte, GTK-Oberfläche und HA-Helfer | Python-CLI, HTTP-Dienst und öffentlicher Collector | Bestehende CLI und Collector weiterverwenden; alte GTK-, HA- und Firmwarekopien bleiben abgelöst. |
| Datenrollen | Positive, schwierige Beispiele, Negative, Ambient, RIR, Distillation getrennt | Aufnahme-, TTS- und Augmentationsordner; kein expliziter Altbestand-Import | `data import`: explizite Rollen, SHA-256, Herkunftsmanifest, getrennte Prüf- und Archivbestände. |
| Datenprüfung | Manifest-Preflight vor dem Training, Prüfung von WAV und MMAP | Kein vollständiger Preflight | `data check` und automatischer Preflight beim Feature-Bau; MMAP-Prüfung vor dem Trainingsprozess. Labelkonflikte und identischer Inhalt in Training/Bewertung blockieren. |
| Bestehende Holdouts | Review-Basis und Bewertungsbestand teilweise gesondert | Inhalts-Split beim Collector-Sync | Explizit importierte Bewertungsrollen bleiben beim Sync erhalten. Alte augmentierte Positive werden archiviert, damit keine Ableitungen von Prüfaufnahmen ins Training gelangen. |
| Zusätzliche Sprache | LibriSpeech Englisch, VoxPopuli Deutsch und Englisch | Mehrere MLS-Sprachen, lokale WAV-Ordner | Download-Presets `librispeech_en`, `voxpopuli_de`, `voxpopuli_en`; bekannte Wakeword-Phrasen werden anhand vorhandener Transkripte ausgeschlossen. Vorhandene WAV-Korpora lassen sich importieren. |
| Sprachclip-Dauer | Download-Sync filterte auf 0,35–8 Sekunden | Keine feste Dauergrenze für zusätzliche Sprach-WAVs | Bestehende gefilterte Lab-WAVs inhaltlich erhalten; neue Downloads haben weiterhin keine entsprechende Dauergrenze. Lange negative Features werden im Backend zufällig abgeschnitten. |
| Sprach-Holdouts | 5 % Validation und 5 % Test, fester Seed | Zusätzliche Sprache nur im Training | `negatives.eval_share: 0.05`, getrennte, deterministische Validation- und Testbestände. |
| Vorgefertigte Negative | Sieben microWakeWord-Archive, auch drei `*_background`-Varianten | Vier Archive | Alle sieben herunterladbar; die vier bisherigen Standardquellen bleiben aktiv. Die älteren drei Background-Archive verwenden 20 statt 10 ms und werden vor dem Training abgelehnt. [Inventur und Qualitätsvergleich](negative-corpora.md). |
| AudioSet und FMA | Breite AudioSet-Auswahl und FMA mit Mengen-/Stundenlimits | Ein AudioSet-Teil und FMA extra small | Vollständige vorhandene WAV-Bestände über `negative_background` importierbar, mit getrennten Features/Holdouts und Gewicht `background_negative`. Der Standarddownload bleibt kleiner; die alten Varianten-Downloader werden nicht wortwörtlich kopiert. |
| Ambient und Raumimpulse | MIT-RIRs und Stichprobe von Hintergrund-WAVs | MIT-RIRs und eigene Hintergrundordner bereits vorhanden | Quellen erhalten; `background` für Augmentation und `negative_background` für negative Trainingsbeispiele ausdrücklich auseinanderhalten. |
| Audio-Augmentation | Im bewährten Profil Positive augmentieren, Negative sauber lassen | Auch Negative und interne Bewertungsfeatures augmentiert | `augmentation.augment_negatives: false` wählbar; Validation und Test erhalten grundsätzlich sauberes Audio. |
| SpecAugment | Zeit- und Frequenzmasken auf den Trainingsfeatures | Masken fest auf null | Vier Parameter unter `training` sind konfigurierbar. Log-Mel-Teacher und Microfrontend-Student bleiben getrennte Feature-Räume. |
| Sehr kurze TTS-Positive | Erfolgreiches Profil filtert unter 0,6 Sekunden | Kein Dauerfilter | `tts.min_positive_duration_seconds: 0.6` wählbar; ausschließlich positive TTS beim Feature-Bau filtern. |
| Student-Architektur | Profile `current`, `current_wider`, weitere Experimente | Festes `esphome_v2_like` | Die beiden bewährten alten Profile sind wählbar; das bisherige Profil bleibt Standard. |
| Kurzes Training | Vollständige Bewertung, LR-Abfall und Frühstopp | Vollständige Bewertung bereits vorhanden, feste Lernrate/Schrittzahl | Optionale `early_stopping_patience`, `early_stopping_min_delta`, `lr_patience`, `lr_factor`; Standard-Patience null erhält das bisherige Verhalten. |
| Verworfene Runden | Angenommenen Trainings-Checkpoint wiederherstellen | Mining verwendete das beste TFLite, Fortsetzung konnte vom verworfenen Checkpoint kommen | Angenommene Gewichte und Restore-Verzeichnis sichern und nach Ablehnung wiederherstellen; auch das aktuelle TFLite zurücksetzen. |
| Unpassende Restore-Formen | Begrenzter automatischer Neuversuch ohne Restore mit Löschen generierter Trainingsdateien | Trainingsfehler meldet Abbruch | Getrennte Architekturprojekte verwenden oder bewusst `train --restart`; kein automatisches Löschen eines Checkpoints nach einer inkompatiblen Profiländerung. |
| Seed-Vergleich | Hypothesen-Queues, getrennte Seed-Kampagnen | Ein optionaler Seed pro Projekt | `experiment prepare/run`: getrennte Projekte auf eingefrorenen Aufnahmen, Mining und Rollout abgeschaltet, gemeinsame Ergebnisdatei. Kein automatischer Übernahmeentscheid. |
| Teacher | Separater PyTorch-Log-Mel-v2-Teacher; frühere Modelle mit vermischten Feature-Räumen waren fehlerhaft | Kein portabler Teacher | `teacher import/train/score`, eigener Log-Mel-Frontend und Checkpoint mit eingebetteten Parametern; optional vorhandene TTS/WAV-Korpora und kompatibles Ausgangsmodell für einen getrennten GPU-Vergleich. Legacy-Konfigurationen mit Microfrontend-MMAP-Negativen werden verweigert. |
| Teacher-Loss | Experimentelle Soft-Label-Distillation; erfolgreicher Juli-Student mit `alpha=0` | Direktes Student-Training | Teacher als separate Prüfstimme. Soft-Label-Loss wird nicht aktiviert; dafür fehlt bislang ein Nachweis auf heutigen Geräten. |
| Prüfung mit STT | Einzelne Diagnose-/Extraktionswerkzeuge | Clip-Extraktion, keine gemeinsame Quellenprüfung | `audit`: Positive, Negative und unbeschriftete Auslösungen mit unabhängiger STT sowie optional Student und Teacher prüfen; keine automatische Labeländerung. |
| Mining | Spätere automatische Mining-Kampagnen verschlechterten das Modell | Optionales Mining, Standard null | Standard null behalten; im Vergleichsprojekt aus. Bestätigte reale Fehlaktivierungen gezielt als Negative sammeln. |
| Bewertung und Ausrollen | Threshold-Sweeps, FAPH-Berichte, ESP-Hardwaretest | 256 Cutoffs, gleiche Prüfaufnahmen, Parity-Gate und Collector-Rollout bereits vorhanden | Vorhandene öffentliche Werkzeuge verwenden. Alte FAPH-Zahlen sind wegen anderer Daten und Betriebsbedingungen keine direkte Messung heutiger Geräte. |
| Leistungsdiagnose | Power-Monitor, GPU-Benchmarks, Feature-/Score-Caches und viele ROCm-Spezialpatches | Ressourcenprüfung und eigene GPU-Prozessverwaltung | [GPU-Benchmark-System](gpu-training-and-benchmarks.md), Graph-Kern, TFRecord/Prefetch und vollständiger Validierungs-Cache sind ergänzt und auf echten Features gemessen. Alte Maschinenpfade und instabile automatische RAM-Caches bleiben abgelöst. |

## Daten übernehmen

Zunächst ein eigenes Projekt erstellen. Das Manifest benennt die Rolle jeder
Quelle ausdrücklich; eine Teacher- oder STT-Vermutung ist keine Freigabe.

```yaml
# sources.yaml; Pfade sind relativ zu dieser Datei oder absolut.
sources:
  - {folder: old/approved/train, role: train_positive}
  - {folder: old/approved/eval, role: eval_positive}
  - {folder: old/confirmed_false_activations, role: train_negative}
  - {folder: old/negative_holdout, role: eval_negative}
  - {folder: old/tts_negative, role: synthetic_negative}
  - {folder: old/voxpopuli_de, role: speech}
  - {folder: old/audioset, role: negative_background}
  - {folder: old/ambient, role: background}
  - {folder: old/rirs, role: rir}
  - {folder: old/pending, role: unlabelled}
  - {folder: old/augmented_positive, role: derived_positive}
```

```bash
wake-word-trainer -p migrated init "Hey Example" --language de
wake-word-trainer -p migrated data import sources.yaml --dry-run
wake-word-trainer -p migrated data import sources.yaml
wake-word-trainer -p migrated data check
```

Weitere Rollen sind `hard_positive` und `synthetic_positive`. Unbeschriftete
Aufnahmen liegen unter `review/unlabelled`, abgeleitete Positive unter
`archive/derived_positive`. Beide gelangen nicht in Trainingsfeatures.
Der Import kopiert Original-WAVs; Mono-Mischung und Resampling auf 16 kHz
erfolgen beim Audio-Lesen. Er regeneriert keine alten MMAPs oder Modelle und
übernimmt keine Secrets. `dataset-import.jsonl` erhält die Quellpfade und
Hashes. `dataset-manifest.jsonl` und `dataset-check.json` dokumentieren den
aktuellen Bestand und seinen Inhalts-Fingerprint.
Unveränderte Hashes werden nur bei identischer Größe, Änderungs-/Statuszeit,
Inode und Gerät wiederverwendet. `data check --rehash` liest alle Inhalte erneut.

Importierte Aufnahmen bleiben beim `fetch` erhalten. Änderungen an deren
Labels müssen daher im importierten Bestand und dessen Herkunftsmanifest
bewusst gepflegt werden; eine Ablehnung im Collector entfernt nur die
Collector-Kopie. Für einen Vergleich ein eigenes, eingefrorenes Projekt ohne
Collector verwenden.

## Bewährtes altes Profil vergleichen

Diese Einstellungen bilden die wesentlichen Stellgrößen des alten kurzen
Regimes ab. Sie sind ein Vergleichsprofil, keine Zusage eines besseren Modells:

```yaml
tts:
  min_positive_duration_seconds: 0.6
augmentation:
  augment_negatives: false
training:
  model_profile: current_wider
  steps: 40000
  eval_step_interval: 500
  early_stopping_patience: 6
  lr_patience: 2
  lr_factor: 0.5
  lr_max_reductions: 3
  min_learning_rate: 0.00001
  time_mask_max_size: 5
  time_mask_count: 2
  freq_mask_max_size: 5
  freq_mask_count: 2
  mining_rounds: 0
deployment:
  enabled: false
```

```bash
wake-word-trainer -p migrated experiment prepare --seed 2026070716 --seed 2026070722
wake-word-trainer -p migrated --downloads shared_downloads experiment run
```

`prepare` kopiert die freigegebenen Aufnahmen und vorhandenen TTS-Clips in
unabhängige Projekte; zusätzliche Korpora bleiben gemeinsame, unveränderte
Quellen. Live-Sync und Sprach-Neugenerierung sind in diesen Projekten aus.
`run` trainiert sie nacheinander. Die besten Gewichte bestimmt weiterhin der
Backend, der Frühstopp erfolgt nach abgeschlossener Bewertung und Sicherung.
`model/training_control.json` dokumentiert Metrik, Schritt, LR-Faktor und Stopp.

Ein Seed allein belegt keine Verbesserung. Modelle mit denselben Prüfaufnahmen
und denselben Cutoffs vergleichen, dann das Parity-Gate und die Hardwareprüfung
verwenden. Der Teacher benötigt zusätzlich eine eigene Bewertung auf den
heutigen Mikrofonkanälen; siehe [Teacher und Quellenprüfung](data-audit.md).
