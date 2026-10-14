# Ein bestehendes Modell ersetzen

Vor dem Wechsel identische Aufnahmen vergleichen:

```bash
wake-word-trainer -p my_project compare --current --model /path/to/reference.tflite
```

Das zugehörige Manifest muss neben jeder TFLite-Datei liegen. Der Bericht
vergleicht beide Modelle an ihren Manifest-Schwellen und an den Schwellen,
die das gleiche Fehlerbudget erlaubt. `--model` ist wiederholbar.

Die gleiche Zahl erkannter Prüfpositive reicht nicht: Auch prüfen, ob dieselben
echten Aufrufe erhalten bleiben und ob eigene sowie Satelliten-Negative öfter
auslösen. Ambient-Audio und Modellhashes bleiben für beide identisch.

Der [Rollout](deployment.md) verlangt keine Regression, eine belegte
Verbesserung, passende Fehlerbudgets und bestandene Parity. Ein Kandidat mit
weniger neuen Fehlalarmen und gleichzeitig verlorenen echten Aufrufen wird
nicht allein wegen der besseren Negativzahl freigegeben.
