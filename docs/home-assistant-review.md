# Fehlaktivierungen direkt in Home Assistant prüfen

Der Collector bleibt die Oberfläche zum Anhören und Bestätigen. Der Trainer
holt ungeprüfte Auslösungen über die Home-Assistant-HTTP-Schnittstelle, führt
unabhängige Spracherkennung und optional Student-/Teacher-Bewertungen durch
und liefert die Ergebnisse an die Collector-Karte zurück. SSH, sudo und ein
Home-Assistant-Administratortoken werden für diesen Ablauf nicht benötigt.

## Einrichtung

Beide öffentlichen Repositories aktualisieren: Wake Word Collector ab 0.5.5
über HACS, Wake Word Trainer ab 0.6.1 auf der Workstation. Nach dem HACS-Update
Home Assistant nach erfolgreicher Konfigurationsprüfung neu starten; die
Frontend-Seite neu laden. Die bisherige Trainer-Adresse und dessen Token in
den Collector-Optionen bleiben gültig. Der Trainer benötigt in `wakeword.yaml`
die Collector-Adresse und dessen vorhandene Token-Datei:

```yaml
collector:
  url: http://homeassistant.local:8123
  token_file: ~/.config/wake-word-trainer/collector.token
```

Auf der Workstation das `segment`-Extra für faster-whisper installieren. Wer
zusätzlich einen Teacher benutzt, benötigt das `teacher`-Extra und ein bereits
geprüftes portables Teacher-Modell. Bei ausdrücklich konfigurierten Profilen
folgendes zum privaten `service.yaml` hinzufügen und den Trainer-Dienst neu
starten, wenn kein Lauf aktiv ist:

```yaml
profiles:
  # Vorhandene Trainingsprofile ebenfalls behalten.
  review:
    label: Fehldetektionen prüfen
    kind: review
    days: 7
    limit: 500
    stt_model: medium
    threads: 4
    # device: bath                  # optional: Collector-Gerätekennung
    # teachers: [teacher/model.pt]  # optional, Pfad relativ zum Projekt
    # models: [deployed/model.tflite]
```

Ohne `models` bewertet der Trainer das vorhandene
`deployment.reference_model`, falls die Datei existiert. Ohne Teacher werden
STT und gegebenenfalls Student trotzdem ausgewertet. Profile können für Räume
und verschiedene Zeiträume vorkonfiguriert werden. Die Nachtplanung akzeptiert
nur Trainingsprofile, keine Prüfung und keinen Benchmark.

## Bedienung

In der Collector-Karte **Aufnahmen prüfen** starten. Der Knopf verwendet das
Profil `review`, andernfalls das erste konfigurierte Prüfprofil. Andere
Prüfprofile lassen sich über die bestehende Profilauswahl und den Startknopf
aufrufen. Fortschritt, Abschluss oder Fehler erscheinen am Trainingssensor.
Die Karte aktualisiert sich auch dann, wenn sich nur die Auswertung geändert
hat und keine neue Aufnahme hinzugekommen ist.

Zu jeder bewerteten Aufnahme zeigt sie den unabhängigen STT-Text, die größte
Student-/Teacher-Antwort und den Prüfvorschlag. **Negativkandidat** ist noch
kein bestätigter Fehlalarm. Anhören, dann **Fehlalarm** oder
**War das Aktivierungswort** wählen; unklare Aufnahmen stehen lassen. Eine
leere Transkription und ein niedriger Teacher-Score beweisen keinen Fehlalarm.
Die Scores unterschiedlicher Modelle sind nicht automatisch kalibriert.

Erst das Bestätigen im Collector macht einen Clip zum Trainingsbeispiel.
Anschließend das normale GPU-Training starten und am gemeinsamen Prüfbestand
mit dem eingesetzten Modell vergleichen. Für jedes Gerät werden außerdem echte
positive Aufrufe vom selben Mikrofon benötigt: Beispiele anderer Räume
belegen dessen Erkennung nicht.

## Daten und Ressourcen

`days` filtert nach **Speicherzeit in Home Assistant**, nicht nach exakt
synchronisiertem Auslösezeitpunkt. Die neuesten passenden Clips kommen zuerst;
`limit` begrenzt den Bestand auf höchstens 2.000. `truncated: true` bedeutet,
dass passende Aufnahmen fehlen und der Zeitraum oder Gerätefilter enger
gewählt werden sollte. Es gibt keine automatische Pagination.

Jeder Lauf bekommt einen neuen privaten Ordner `review/run_…` mit WAV-Snapshot,
Prüfsummen, Herkunft und Auditbericht. Diese Dateien werden nicht in
`recordings/train` oder `recordings/eval` importiert. `review.json` enthält die
letzte Auswertung. Audio und Transkripte bleiben außerhalb von Git. Der
Snapshot-Ordner ist nur für den Besitzer zugänglich, WAVs und Berichte haben
Dateimodus 0600. Alte Snapshots bei Bedarf selbst löschen.

Die Prüfung nutzt die CPU, begrenzt ihre Threads und pausiert keine GPU-Dienste.
Sie kann bei einer anderweitig belegten GPU laufen; ein aktiver Lauf desselben
Trainer-Dienstes oder ein Firmware-Rollout blockiert einen zweiten Start.
RAM- und CPU-Prüfung bleiben wirksam. Das eigentliche Student-Training nutzt
weiter die konfigurierte GPU. Ein Prüfabschluss bietet kein altes oder neues
Modell an und löst keinen Rollout aus.

## HTTP-Schnittstellen

Der Collector akzeptiert für beide neuen GET-Routen ausschließlich seinen
vorhandenen `X-Wakeword-Token`. Die Routen ändern keine Labels:

| Route | Inhalt |
| --- | --- |
| `/api/wake_word_collector/export/<slug>/review` | `clips`, `labels_confirmed: false`, `total`, `truncated`; Filter `device`, `created_after` mit Zeitzone und `limit` |
| `/api/wake_word_collector/export/<slug>/review/<category>/<device>/<filename>` | WAV aus `triggers`, `needs_review` oder `rejected_transcript` |

Bestätigte Positive und Negative bleiben in den bisherigen getrennten
Trainings-Exporten. Der Trainer akzeptiert nur Audioverweise genau innerhalb
dieser Review-Route, prüft vorhandene SHA-256-Werte und folgt keinen HTTP-
Weiterleitungen mit dem Token. Nach einer Verschiebung oder Bearbeitung kann
ein Snapshot-Abruf fehlschlagen; dann einen neuen Prüfstart verwenden.

`GET /v1/review` am Trainer benötigt dessen vorhandenen Bearer-Token und
liefert den privaten Detailbericht. `/v1/status` enthält nur eine Zusammenfassung
unter `review` sowie `review_profiles`. Die Collector-Karte holt Detailtexte
über den angemeldeten WebSocket-Zugang; Transkripte werden nicht als neue
Sensorattribute an den Recorder gegeben. Prüfergebnisse werden anhand des
Audiohashes zugeordnet: Nach einem Schnitt passt ein altes Ergebnis nicht mehr.

Für einen gezielten lokalen Aufruf derselben HA-Schnittstelle:

```bash
wake-word-trainer -p my_project review --device bath \
  --since '2026-10-14T00:00:00+02:00' --stt-model medium \
  --teacher teacher/model.pt
```

Dieser Abruf benötigt ebenfalls nur den Collector-Token. Eine 404-Meldung
über eine fehlende Prüfschnittstelle bedeutet, dass der Collector zuerst
aktualisiert werden muss.
