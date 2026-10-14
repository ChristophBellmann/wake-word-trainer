# Bessere Modelle mit passenden Daten

Ein Modell muss echte Aufrufe erhalten und Sprache ohne Aufruf ignorieren.
Mehr Daten helfen nur, wenn Labels, Quelle und Prüfrollen passen.

## Eigene Positive und Negative

Echte positive Aufrufe aus denselben Räumen, Entfernungen und Mikrofonkanälen
sammeln. Synthetische Stimmen ergänzen diese Vielfalt, ersetzen sie aber nicht.
Bestätigte TV-, Gesprächs- oder Geräteansagen ohne Aktivierungswort bilden
besonders passende Negative für den jeweiligen Alltag.

Der Collector liefert tatsächliche Satelliten-Fehlalarme als eigene Quelle
`satellite_negative` mit Standardgewicht 4. Lokale negative WAV-Ordner gehören
zu `own_negative`. Die Trennung verhindert, dass ein sehr großer lokaler
Bestand wenige echte Satellitenfehler verdrängt. Der Bericht zählt beide
Quellen getrennt; ein Rollout darf die Satelliten-Negativfehler nicht erhöhen.

## Teacher und unabhängige STT

Die [Quellenprüfung](data-audit.md) zeigt Widersprüche und bewertet schwierige
Negative. Ein Teacher kann gezielt auf aktuelle Mikrofonaufnahmen angepasst
werden. Seine Log-Mel-Features bleiben getrennt vom Microfrontend des Students.
Eine stärkere STT kann Geräteansagen oder Gespräche erkennen; ein nicht
gefundenes Aktivierungswort allein beweist noch keinen Fehlalarm.
Die [HA-Prüfung](home-assistant-review.md) verbindet diese Ergebnisse mit den
Aufnahmen und deren Bestätigung.

## Sprache, Räume und schwierige Beispiele

```bash
wake-word-trainer -p my_project download --speech mls_de
```

Multilingual LibriSpeech liefert zusätzliche Sprache; neben Deutsch sind
`mls_fr`, `mls_nl`, `mls_es`, `mls_it`, `mls_pt` und `mls_pl` verfügbar.
Die Auswahl unter `negatives.speech_presets` übernehmen. Eigene Raumgeräusche
und Raumantworten unter `augmentation.background_folders` und `rir_folders`
verbessern die Augmentierung, sofern sie nicht in den Prüfbestand gelangen.

`training.mining_rounds` sammelt nach einer Runde falsch ausgelöste Fenster
negativer **Trainingsdaten**. Mining separat vergleichen und dessen Gewichtung
begrenzen; es kann die Qualität auch verschlechtern. Prüfclips bleiben verboten.

Für ein Zweiwort-Modell ist das zweite Wort allein kein vollständiger Aufruf.
Teacher-Ziele können zusätzliche Prüfaufgaben abdecken, müssen aber ausdrücklich
von Student-Labels getrennt bleiben. Seeds, Sampling-Anteile und unverändertes
Prüfaudio machen den Vergleich nachvollziehbar.
