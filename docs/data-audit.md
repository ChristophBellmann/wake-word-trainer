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
