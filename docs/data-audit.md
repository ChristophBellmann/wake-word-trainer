# Positive, Negative und Fehlaktivierungen prüfen

`audit` prüft vorhandene Aufnahmen mit einer unabhängigen lokalen
Spracherkennung und optional mit den Wakeword-Modellen. Der Befehl verändert
keine Labels, holt keine neuen Aufnahmen und startet kein Training oder OTA.
Zunächst `fetch` ausführen, wenn der Collector-Bestand aktualisiert werden soll.
Benötigt werden das Extra `segment` und für TFLite-Bewertungen zusätzlich `train`.

```bash
wake-word-trainer -p my_project fetch
wake-word-trainer -p my_project audit --stt-model small \
  --model deployed/model.tflite --model export/my_word.tflite \
  --unlabelled-folder activation_clips --output audit
```

Standardmäßig werden die eigenen Aufnahmen unter `recordings/` geprüft,
einschließlich Training, Bewertung und schwieriger Beispiele. Zusätzliche
Quellen kommen über `--positive-folder`, `--negative-folder` und
`--unlabelled-folder` hinzu; jede Option ist wiederholbar. Ordner sind relativ
zum Projekt, absolute Pfade sind ebenfalls erlaubt. Gleichlautende WAV-Inhalte
werden über SHA-256 nur einmal verarbeitet; alle Quellen und Labels bleiben im
Bericht sichtbar. Gegenläufige Labels desselben Inhalts werden markiert.

`--phrase` ergänzt erlaubte Schreibweisen der Spracherkennung. Standardmäßig
gelten das Aktivierungswort und `tts.phrases`. Die Erkennung erhält keinen
Prompt mit dem erwarteten Aktivierungswort. `--stt-model` erlaubt ein größeres
Whisper-Modell oder einen lokalen Modellordner; `--threads` begrenzt die
CPU-Threads (Standard: vier). Eine fehlende STT-Übereinstimmung beweist keinen
Fehlalarm, besonders bei stark gefilterten Mikrofonkanälen.

## Bericht und Prüfung

`clips.jsonl` enthält pro WAV-Inhalt Quellen, Transkript, Wortzeiten,
gefundene Phrasen, Modellspitzen und Prüfbefunde. `summary.json` enthält die
Anzahlen und die verwendeten Einstellungen. Beides gehört zum privaten
Projekt: Auch Transkripte sind Sprachdaten und gehören nicht ins öffentliche Git.
Die Konsole meldet nur Fortschritt und Anzahlen.

| Befund | Bedeutung |
| --- | --- |
| `conflicting_labels` | Derselbe Inhalt ist positiv und negativ beschriftet. |
| `positive_without_stt_match` | STT bestätigt die positive Beschriftung nicht; anhören. |
| `negative_with_stt_match` | STT findet das Aktivierungswort in einem Negativbeispiel. |
| `hard_negative` | Mindestens ein Modell überschreitet bei einem Negativbeispiel seine Manifest-Schwelle. |
| `missed_positive` | Mindestens ein Modell verpasst ein positives Beispiel an seiner Manifest-Schwelle. |
| `wake_word_candidate` | Unbeschrifteter Clip enthält laut STT die konfigurierte Phrase. |
| `negative_candidate` | Unbeschrifteter Clip enthält erkannte Sprache ohne passende Phrase. |
| `unclear_audio` | Unbeschrifteter Clip liefert keine verwertbaren Wörter. |
| `processing_error` | Verarbeitung gescheitert; niemals als negatives Ergebnis auslegen. |

Alle Befunde verlangen Anhören vor einer Labeländerung. Die Modellspitze ist
das Maximum des gleitenden Wahrscheinlichkeitsmittels mit derselben
Vorlaufstille wie in der Offline-Bewertung. Modellhash, Manifest-Schwelle und
Fenstergröße sind im Bericht gebunden. Das ist ein Offline-Vergleich;
Mikrofonkanal, Verstärkung, Firmwarestand und Gerätestufe müssen für einen
Vergleich mit einer tatsächlichen Auslösung ebenfalls passen.

Ein erneuter Aufruf verwendet erfolgreiche Ergebnisse nur bei unveränderten
Audioinhalten und Prüfeinstellungen wieder. Neue Modellgewichte, Schwellen,
Phrasen oder STT-Modellnamen verlangen eine neue Verarbeitung. Fehler werden
erneut versucht. Während einer Prüfung niemals denselben Ausgabeordner mit
einem zweiten Prozess verwenden. Bei Austausch eines STT-Modells unter
demselben lokalen Pfad einen neuen Ausgabeordner verwenden.

## Daraus Training vorbereiten

Bestätigte Fehlaktivierungen im Collector als negativ markieren. Echte
Aufrufe positiv freigeben, unklare Clips dort zur Prüfung stehen lassen.
Danach `fetch` und `run --prepare-only` ausführen. Der feste Inhalts-Split
bleibt erhalten; `prospective_split` im Bericht zeigt lediglich die
Zuordnung des Inhalts, auch für bislang unbeschriftete Clips. Eine Kopie in
einen Trainingsordner darf keine Prüfaufnahme ins Training zurückholen.

Der aktuelle Trainer trainiert microWakeWord direkt und nutzt
[Mining-Runden](better-models.md#mining-rounds) für schwierige Negative.
Er benötigt keinen Teacher. Ein separat vorhandener Teacher kann als
zusätzliche Prüfstimme dienen, wenn er auf den heutigen Mikrofonaufnahmen
nachweislich zuverlässiger ist. Seine Fehlentscheidungen automatisch als
Labels zu übernehmen würde sie im nächsten Modell festigen. Ein Teacher-
Neutraining sollte deshalb auf geprüften Labels basieren und auf getrennten
heutigen Satellitenaufnahmen bewertet werden.

## Portabler Teacher

Das zusätzliche Extra `teacher` installiert PyTorch. Der Teacher hat ein
eigenes Log-Mel-Frontend; Microfrontend-MMAPs des Students dürfen ihm nicht
übergeben werden. Beim Import werden Architektur, Gewichte und die
Feature-Parameter aus der ursprünglichen YAML gemeinsam gespeichert. Die
alte Konfiguration wird danach nicht mehr benötigt.

```bash
wake-word-trainer -p my_project teacher import /private/lab/best_model.pt \
  --config /private/lab/teacher.yaml --output teacher/legacy.pt
wake-word-trainer -p my_project audit --teacher teacher/legacy.pt \
  --unlabelled-folder activation_clips
wake-word-trainer -p my_project teacher train --epochs 30 --seed 1337
wake-word-trainer -p my_project teacher score --model teacher/model.pt \
  --folder activation_clips --output teacher/scores.json
```

Importiert werden ausschließlich kompatible Log-Mel-v2-Gewichte. Eine
Legacy-Konfiguration mit `negative_mmap_roots` wird verweigert: Diese Modelle
können mit unterschiedlichen Feature-Räumen trainiert worden sein. Ein
Schwellwert aus dem alten Lab muss auf aktuellen Aufnahmen erneut geprüft
werden. `teacher score` schreibt Maximum und 95-%-Quantil gleitender Fenster;
das sind rohe Teacher-Scores, keine kalibrierten Labelwahrscheinlichkeiten.

`teacher train` ist ein separater Baseline-Lauf auf den freigegebenen
Positiven und Negativen in `recordings/train` und `recordings/negative/train`.
Bewertungsaufnahmen bleiben aus dem Optimierer und bestimmen den besten
Checkpoint sowie den Frühstopp. Positive und Bewertungsclips liefern das letzte
1,5-Sekunden-Fenster; längere Trainingsnegative ein reproduzierbares zufälliges
Fenster. Die anschließende Bewertung prüft alle gleitenden Fenster.
`--device gpu` verlangt GPU, `auto` bevorzugt GPU und `--device cpu` erzwingt
CPU. Die Log-Mel-Aufbereitung bleibt auf CPU; Modell und Optimierer laufen
auf dem gewählten Gerät. GPU-Arbeit wird mit Student und Benchmarks serialisiert.

Der Baseline-Lauf nutzt eigene Aufnahmen. Ein zusätzlicher Vergleich kann die
bereits vorbereiteten positiven/negativen TTS, schwierigen Positive und externe
WAV-Sprach-/Hintergrundkorpora einbeziehen:

```bash
wake-word-trainer -p comparison teacher train --device gpu --epochs 40 \
  --include-corpus --negative-limit 6000 --balanced-sources \
  --initial-model teacher/legacy.pt
```

`--initial-model` ist optional und verlangt identische Log-Mel-Parameter.
`--negative-limit` begrenzt jedes externe Korpus nach dem Trainings-/Holdout-
Split. Derselbe Inhalts-Split wie beim Student hält dessen externe Prüfclips
aus dem Teacher-Optimierer. Positive TTS beachten den Dauerfilter; lange
Trainingsnegative liefern ein reproduzierbares zufälliges Fenster. Eigene
Bewertungsaufnahmen bestimmen weiter den besten Checkpoint. Quellen und
Anzahlen stehen privat in `teacher/training-sources.json` und im Checkpoint.

`--balanced-sources` verwendet die Quellanteile aus `training.weights` auch
beim Teacher. Damit verdrängen große Korpora weder eigene Positive noch die
wenigen bestätigten Satelliten-Negative. Die Klassengewichtung berücksichtigt
die tatsächlich gesampelten positiven/negativen Anteile. Ohne diese Option
zieht der Teacher gleichmäßig aus allen Trainingsclips; ein größeres Korpus
verändert dann auch die Anteile. Beide Wege sind getrennt vergleichbar.

Beim Scoring teilen überlappende Fenster ihre STFT-Berechnung. Alle bisherigen
Fenster bleiben erhalten; ein nicht am Frame-Raster liegendes Schlussfenster
wird gesondert berechnet. Die Parität ist auch für ungerade Cliplängen geprüft.

Microfrontend-MMAPs bleiben ausgeschlossen. Audio-Augmentation und das
vollständige alte Teacher-Regime werden damit noch nicht reproduziert. Der
Lauf aktiviert weder Teacher-Loss im Student noch Rollout. Erst der Vergleich
auf heutigen Satellitenaufnahmen entscheidet, ob der neue Teacher hilft.

Für Herkunft, sichere Datenübernahme und getrennte Seed-Läufe:
[Unterschiede und Übernahme aus dem Wakeword Lab](migration-from-wakeword-lab.md).
