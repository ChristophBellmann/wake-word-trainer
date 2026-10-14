# Negativkorpora aus dem Wakeword Lab

Die vorgefertigten Korpus-Negative werden bereits verwendet: `speech`,
`dinner_party` und `no_speech` gehören zu den Standardquellen des Students.
`dinner_party_eval` ist ausschließlich Ambient für Validation und Prüfung,
mit Sampling-Gewicht null. Zusätzliche WAV-Korpora kommen getrennt hinzu.
Diese Microfrontend-MMAPs gehören nicht in den Log-Mel-Teacher.

## Was tatsächlich vorhanden ist

Inventur vom 14.10.2026. Die Zahlen zählen MMAP-Einträge, keine individuellen
Sprecher oder unabhängigen Gespräche. Wiederholte Ausschnitte derselben
Aufnahme sind nicht unabhängige Beobachtungen.

| HF-Archiv | Trainings-Einträge im alten Bestand | Bisherige neue Läufe | Standardgewicht |
| --- | ---: | --- | ---: |
| `speech` | 350.535 | als Negative enthalten | 10 |
| `dinner_party` | 58.307 | als Negative enthalten | 10 |
| `no_speech` | 238.441 | als Negative enthalten | 5 |
| `dinner_party_eval` | 0; 4 Validation- und 10 Prüfaufnahmen | ausschließlich Ambient-Holdout | 0 |
| `speech_background` | 630.158 | nicht enthalten; alte 20-ms-Features | — |
| `dinner_party_background` | 29.153; zusätzlich Ambient-Holdouts | nicht enthalten; alte 20-ms-Features | — |
| `no_speech_background` | 228.518 | nicht enthalten; alte 20-ms-Features | — |

Die ersten vier entpackten Archive belegen hier ungefähr 25 GiB. Die drei
zusätzlichen belegen im alten Bestand weitere rund 17 GiB. Alle sieben
Archive lassen sich herunterladen; das bedeutet nicht, dass alle mit dem
aktuellen Frontend kompatibel sind.

Die sechs alten Trainingsquellen umfassen zusammen **1.535.112 Einträge**;
die drei kompatiblen aktuellen HF-Trainingsquellen **647.283 Einträge**.
Diese Summe ist keine Zahl einzigartiger Audios: etwa die beiden CHiME-
Fassungen enthalten überlappende Originalaufnahmen bei anderem Frontend.

Die [HF-Quellenbeschreibung](https://huggingface.co/datasets/kahrendt/microwakeword/blob/main/README.md)
nennt für die älteren Background-Archive 20-ms-Schritte und beschreibt
CHiME6/DipCo-Gespräche, LibriSpeech/VOiCES sowie FMA/FSD50K/WHAM. Der Trainer
und die ESPHome-Modelle benötigen 10-ms-Features. Im lokalen DipCo-Bestand
enthält die ältere Background-Fassung für dasselbe Audio halb so viele
Frames wie `dinner_party_eval`. Die Auswahl eines vorhandenen alten
`*_background`-Archivs wird deshalb vor dem Training mit einem konkreten
Fehler abgelehnt. Ein Float-Cast oder Verdoppeln der Frames regeneriert
keine passenden Microfrontend-Features. Dafür die Original-WAVs verwenden.

Unter dem Namen „Coffee Talk“ fand sich in den geprüften Lab-Skripten,
Manifesten und Downloadquellen kein eigener Datensatz. `dinner_party` ist
anhand seiner MMAP-Namen als CHiME-Gesprächsbestand identifiziert. Eine
Gleichsetzung mit einem weiteren namensähnlichen Korpus ist nicht belegt.

## Download und aktive Auswahl

```bash
wake-word-trainer -p project --downloads shared_downloads download \
  --negative-set speech --negative-set dinner_party \
  --negative-set no_speech --negative-set dinner_party_eval
```

```yaml
negatives:
  feature_sets: [speech, dinner_party, no_speech, dinner_party_eval]
training:
  weights:
    speech: 10
    dinner_party: 10
    no_speech: 5
```

Der Download legt Daten ab; `negatives.feature_sets` bestimmt die Auswahl.
Fehlende konfigurierte Trainingsquellen werden im Log ausdrücklich gemeldet.
`training_parameters.yaml` dokumentiert die tatsächlich verwendeten Pfade,
Gewichte und Datenrollen. Dort nachprüfen, statt aus vorhandenen ZIP-Dateien
oder einem alten YAML-Manifest auf die Nutzung zu schließen.

`dinner_party_eval` bleibt bei vorhandenem Download für die Bewertung erhalten,
auch wenn die drei Trainingsquellen aus `feature_sets` entfernt werden. Das
ermöglicht einen Vergleich auf unverändertem Ambient-Audio.

## Welche alten Hilfsmittel weitergeführt werden

| Altes Werkzeug | Heutiger Ablauf |
| --- | --- |
| `scripts/setup/setup_sync_datasets.py` | `download --negative-set`, Sprach-Presets, `data import`, `features`; vorhandene kompatible MMAPs direkt wiederverwenden |
| `scripts/setup/setup_fetch_notebook_backgrounds.py` | vorhandene AudioSet-/FMA-WAVs als `negative_background` importieren; eigene Features mit 10 ms erzeugen |
| `scripts/train_from_config.sh` | `wakeword.yaml` und `train`/`run`; tatsächlich erzeugte Backend-Konfiguration nachsehen |
| `docs/training_dataset.md` | Rollen weiterhin trennen: Korpus-Negative, WAV-Augmentation, Ambient-Holdout und RIR |

In den eingefrorenen Vergleichsläufen sind bereits 15.839 Sprach-WAVs aus
VoxPopuli Deutsch/Englisch und LibriSpeech sowie 41.235 AudioSet-/FMA-WAVs
übernommen. Ihre inhaltlich getrennten Training-/Validation-/Testseiten
bleiben erhalten. Raumimpulse und die Hintergrundstichprobe für
Audio-Augmentation erfüllen andere Rollen.

## Vergleich mit und ohne HF-Korpus-Negative

Die frühere [Ambient-Skalierung wurde korrigiert](evaluation.md#korrektur-der-mmap-skalierung-am-14102026).
Frühere FAPH-Zahlen und davon abhängige Parity-Freigaben müssen neu berechnet
werden. Eigene WAV-Trefferzahlen waren davon nicht betroffen.

Der kontrollierte Vergleich verwendet das offizielle Modellprofil, Batch 256,
GPU, optimierten TFRecord-Eingang, dieselben eingefrorenen Features, Seeds
`2026070716` und `2026070722`, identischen Frühstopp sowie unveränderte
47 positive und 319 negative Prüfclips, darunter 14 Satelliten-Negative.
Das Ambient-Prüfaudio bleibt dasselbe. Kein Mining, Collector-Sync oder
Firmware-Rollout erfolgt während der Vergleiche.

Einfaches Entfernen der drei Quellen verändert auch die Sampling-Verteilung:
Die verbleibenden Gewichte werden normalisiert, also steigen die Anteile
eigener Aufnahmen und TTS. Deshalb gibt es zusätzlich einen Vergleich mit
Ersatz durch die bereits vorhandenen WAV-Sprach-/Hintergrundkorpora:
`speech_extra: 25`, `background_negative: 10` statt jeweils 5. Die wegfallende
HF-Gewichtsmasse 25 wird dadurch ersetzt; positive/negative Gesamtsampling-
Masse und Klassengewichte bleiben wie beim Ausgangsmodell. Das prüft den
Nutzen gegenüber diesen konkreten Ersatzkorpora.

### Ergebnisse mit korrigierter Inferenz

Vier neue GPU-Trainingsläufe wurden mit den beiden bereits vorhandenen
HF-Ausgangsmodellen verglichen. Alle sechs Modelle und das laufende Modell
wurden anschließend mit korrekter MMAP-Skalierung und ganzzahligen
8-Bit-Fenstersummen auf demselben Audio neu bewertet.

Bei identischer Schwelle **230/255**:

| Variante | Seed-Ende | Positive / 47 | Eigene Negativfehler / 319 | Satellitenfehler / 14 | Ambient-Fehler/h |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF-Standardquellen | 16 | 46 | 3 | 3 | 0,562 |
| HF entfernt | 16 | 45 | 6 | 4 | 0,750 |
| HF durch WAV-Korpora ersetzt, Sampling-Masse gleich | 16 | 43 | 4 | 3 | 0,187 |
| HF-Standardquellen | 22 | 41 | 2 | 2 | 0,000 |
| HF entfernt | 22 | 45 | 6 | 4 | 1,312 |
| HF durch WAV-Korpora ersetzt, Sampling-Masse gleich | 22 | 44 | 7 | 5 | 0,000 |
| Laufendes Modell als Referenz | — | 46 | 5 | 5 | 0,000 |

Die gleiche Zahl erkannter Wakewords ist die bessere Vergleichsbasis als
allein die gleiche numerische Schwelle. Bei mindestens **46/47 Treffern**,
jeweils mit der höchsten noch möglichen 8-Bit-Schwelle:

| Variante | Seed-Ende | Ambient-Fehler/h | Satellitenfehler / 14 |
| --- | ---: | ---: | ---: |
| HF-Standardquellen | 16 | 0,562 | 3 |
| HF entfernt | 16 | 3,562 | 7 |
| HF durch WAV-Korpora ersetzt | 16 | 7,311 | 8 |
| HF-Standardquellen | 22 | 4,499 | 7 |
| HF entfernt | 22 | 14,810 | 6 |
| HF durch WAV-Korpora ersetzt | 22 | 10,873 | 7 |
| Laufendes Modell als Referenz | — | 0,000 | 4 |

**Entscheidung:** Die drei kompatiblen HF-Quellen bleiben enthalten. Auf
beiden Seeds senken sie bei vergleichbarer Erkennung die Ambient-Fehler
gegenüber Entfernen und diesen konkreten Ersatzkorpora. Es gibt zugleich
Zielkonflikte und deutliche Seed-Streuung. Keines dieser sechs Modelle
verbessert die laufende Referenz unter dem festgelegten Budget bei gleicher
Erkennung. Kein Modell wurde ausgerollt.

Bei einem beobachteten Budget von 0,1 Fehler/h erkennt HF-Seed 16 nur 43/47,
HF-Seed 22 41/47. Ohne HF sind es 45/47 beziehungsweise 0/47: Beim zweiten
Seed erreicht erst 255/255 das Budget, dort darf kein Wakeword auslösen.
Die zuvor durch Float-Rundung angezeigten Treffer oberhalb von 1,0 sind
entfernt. Die Ersatzvarianten erreichen 43/47 beziehungsweise 44/47.

Das Ambient-Audio umfasst 5,334 Stunden. Ein Fehler entspricht rund
0,187 Fehlern/h; null beobachtete Fehler belegen daher noch keine stabile
Alltagsrate unter 0,1/h. Diese kleinen, bereits zur Auswahl verwendeten
Holdouts sind kein unabhängiger Alltagstest. Dafür weitere bestätigte
Collector-Negative und einen neuen Prüfbestand sammeln.

## Korpusgröße und TFRecord-Pool unterscheiden

`training.tfrecord_examples: 65536` begrenzt den einmal gezogenen statischen
Eingangspool, nicht den Downloadbestand. Ein langer Lauf liest diesen Pool
wiederholt; SpecAugment zieht dabei neue Masken. Mehr als eine Million
verarbeitete Beispiele bedeuten deshalb nicht mehr als eine Million
unterschiedliche Korpus-Clips. Der Standardpfad `prefetch` zieht fortlaufend
aus den vollständigen Quellen, ist auf diesem Bestand aber deutlich langsamer.

Ein zusätzlicher Vergleich mit **1.048.576 gezogenen Pool-Beispielen** auf
denselben beiden Seeds prüft diese mögliche Begrenzung separat. Quellen,
Gewichte, Architektur, Features, Batchgröße und Prüfaudio bleiben dabei
identisch. Der Pool ist hier rund 32 GiB groß und wird als Datenträgerdatei
gestreamt. Beide GPU-Läufe sind abgeschlossen, bei Cutoff 230/255:

| Pool | Seed-Ende | Positive / 47 | Eigene Negativfehler / 319 | Satellitenfehler / 14 | Ambient-Fehler/h |
| --- | ---: | ---: | ---: | ---: | ---: |
| 65.536, HF-Standard | 16 | 46 | 3 | 3 | 0,562 |
| 1.048.576, gleiche Quellen | 16 | 46 | 8 | 6 | 0,000 |
| 65.536, HF-Standard | 22 | 41 | 2 | 2 | 0,000 |
| 1.048.576, gleiche Quellen | 22 | 44 | 5 | 3 | 0,000 |

Der größere Pool senkt auf Seed 16 die beobachteten Ambient-Fehler und erhöht
auf Seed 22 die Erkennung. Auf beiden Seeds nehmen jedoch die eigenen und
Satelliten-Fehler zu. Gegenüber dem laufenden Modell gewinnt keiner dieser
Läufe den festgelegten Gesamtvergleich. **Die Dienstvorgabe bleibt 65.536.**

Der einmalige Pool-Bau dauerte etwa 490 beziehungsweise 644 Sekunden. Die
Läufe stoppten nach 13.500 beziehungsweise 5.000 Schritten. Der größere Pool
passte nicht vollständig in den verfügbaren Dateicache und verursachte mehr
Input-Wartezeit. Diese Zeiten sind keine isolierte Durchsatzmessung: Während
Teilen des ersten Laufs wurde parallel das neue native SDK vorbereitet.
Die Modellqualität wurde weiterhin offline auf identischem Audio gemessen.
Die temporären 32-GiB-Dateien wurden nach der Messung entfernt; Signaturen,
Prüfsummen, Berichte und Modelle bleiben privat erhalten.
