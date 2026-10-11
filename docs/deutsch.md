# Deutsch

Ein Aktivierungswort-Modell für ESPHome-Sprachsatelliten trainieren, aus den
**eigenen Aufnahmen** (gesammelt mit dem
[Wake Word Collector](https://github.com/ChristophBellmann/ha-wake-word-collector))
und synthetischer Sprache, und vor dem Flashen ehrlich messen, wie gut es bei
der eigenen Stimme erkennt.

```bash
wake-word-trainer -p hey_jarvis init "Hey Jarvis" --language de \
  --collector-url http://homeassistant.local:8123 --token-file ~/.config/wake-word-trainer/hey_jarvis.token
wake-word-trainer -p hey_jarvis download --voice de_DE-thorsten-medium --voice de_DE-kerstin-low --data
# wakeword.yaml: tts.voices, tts.phrases, tts.negative_phrases eintragen
wake-word-trainer -p hey_jarvis run
```

Der Trainer holt die im Collector angenommenen Aufnahmen und teilt sie fest in
Training und Bewertung. Er erzeugt synthetische Beispiele und ähnlich klingende
Gegenbeispiele und trainiert mit microWakeWord. Bewertet wird nur auf den
zurückgehaltenen eigenen Aufnahmen. Die Schwelle wird nach dem erlaubten Budget
an Fehlauslösungen pro Stunde gewählt. Am Ende liegen `.tflite` und das
ESPHome-Manifest unter `export/`. Nicht erkannte Aufnahmen werden aufgelistet:
anhören und, wenn falsch, im Collector verwerfen.

Schwierige Beispiele dürfen den Bewertungsbestand nicht ins Training
zurückbringen: Liegt derselbe WAV-Inhalt auch unter `recordings.hard_folders`,
bleibt eine bereits zur Bewertung zugeordnete Aufnahme ausschließlich dort.
`fetch` entfernt frühere doppelte Trainingskopien; Quelldateien bleiben erhalten.

## Aktivierungswörter aus langen Aufnahmen

Mit dem Extra `segment` und `extraction.enabled: true` schneidet der
Collector längere Mikrofonaufnahmen automatisch in einzelne Aktivierungswörter.
Phrase und Varianten kommen aus der HA-Integration; Sprache und Erkennungsmodell
sind konfigurierbar. Das Original bleibt in HA, die Ausschnitte warten dort auf
Freigabe. Der bestehende Trainingslauf und dessen GPU bleiben unberührt.

## Home Assistant

`wake-word-trainer serve` läuft auf dem Trainingsrechner. Im Wake Word
Collector trägt man `http://<Trainingsrechner>:10701` und das Token ein; dann
startet und verfolgt man das Training aus Home Assistant, und das fertige
Modell rollt Home Assistant auf die Satelliten aus. Andere GPU-Dienste (LLM,
TTS) pausiert der Trainer während eines Laufs und startet sie danach wieder.
Ist der Trainingsrechner aus, meldet der Collector den Trainer als nicht
erreichbar; das ist normal.

Optional trainiert der Dienst nachts im Leerlauf neue Aufnahmen und spielt
ein neues Modell nur dann automatisch aus, wenn es auf denselben
zurückgehaltenen Aufnahmen mindestens so gut ist wie das laufende.

Die ausführliche Dokumentation auf den übrigen Seiten ist englisch.
