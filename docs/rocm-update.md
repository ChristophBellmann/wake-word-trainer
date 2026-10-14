# ROCm und MIOpen aktualisieren

Ein neuer GPU-Stack wird zuerst getrennt verglichen. Die
[AMD-Kompatibilitätsmatrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
und der [TheRock-Entwicklungsstand](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
beschreiben unterschiedliche Freigaben. Am 14.10.2026 führt TheRock `gfx1031`
für die RX 6700 XT als gebaut, getestet und releasefähig. Die verwendeten
Frameworks benötigen trotzdem einen eigenen Funktionstest.

## Geprüfter Teacher-Vergleich

Dieselbe RX 6700 XT, zwei Wiederholungen, im zweiten Durchgang umgekehrte
Reihenfolge, 20 Aufwärm- und 100 Messschritte. Eingang und portables
`temporal_v1`-Modell waren identisch: 64 echte Log-Mel-Fenster aus eigenen
Prüfaufnahmen, für größere Batches wiederholt. Dropout war für den Vergleich
null. Datenvorbereitung und STT sind in diesen Werten nicht enthalten.

| Umgebung | PyTorch | MIOpen | Batch | Beispiele/s, Median |
| --- | --- | --- | ---: | ---: |
| Eigenbau, ROCm 7.11 | 2.11 dev | 3.5.1 | 64 | 7.475 |
| AMD-Pakete, ROCm 10.1 | 2.12 | 3.6.1 | 64 | 12.613 |
| Eigenbau, ROCm 7.11 | 2.11 dev | 3.5.1 | 256 | 9.826 |
| AMD-Pakete, ROCm 10.1 | 2.12 | 3.6.1 | 256 | 14.674 |

Der neue Stack ist bei Batch 256 rund **49 % schneller**, bei Batch 64 rund
69 %. Framework und Runtime ändern sich gemeinsam; der Gewinn lässt sich
nicht allein MIOpen zuschreiben. Das Profil zeigt tatsächliche
`miopen_convolution`-/`miopen_depthwise_convolution`-Aufrufe und passende
Bibliothekspfade. Kein `HSA_OVERRIDE_GFX_VERSION` war nötig. Die größte
Logit-Abweichung zwischen den GPU-Umgebungen lag bei etwa `6,7e-6`; beide
stimmen ebenso eng mit der CPU überein. Der Rechenkernvergleich belegt keine
bessere Erkennungsqualität.

## Separate AMD-Python-Umgebung

Nach der [AMD-Release-Anleitung](https://github.com/ROCm/TheRock/blob/main/RELEASES.md),
mit den tatsächlich geprüften Versionen:

```bash
python3.12 -m venv teacher-rocm10
teacher-rocm10/bin/python -m pip install \
  --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  --extra-index-url https://pypi.org/simple \
  'torch[device-gfx1031]==2.12.0+rocm10.1.0' 'numpy==1.26.4'
teacher-rocm10/bin/python -m pip install wake-word-trainer
```

Die genaue Torch-Version verhindert eine andere Auswahl über den zusätzlichen
PyPI-Index. Die GPU-Pakete bringen ihre passenden SDK-Bibliotheken mit. Der
Trainer erkennt `rocm-sdk-core` und `rocm-sdk-libraries` ohne GPU-Initialisierung
und mischt nicht automatisch `/opt/rocm`, dessen RCCL-Preload oder alte
TensorFlow-Workarounds hinzu. Ein ausdrücklich gesetztes `ROCM_PATH` bleibt
eine bewusste Auswahl. Für die neue Umgebung keine alte Shell-Aktivierung
oder deren `LD_LIBRARY_PATH`-/`LD_PRELOAD`-Werte übernehmen.

Die öffentliche Teacher-CLI ist mit den neuen Paketen auf eigenen Aufnahmen
geprüft. Student und Teacher dürfen getrennte Python-Umgebungen verwenden.

## Student separat prüfen

microWakeWord benötigt TensorFlow und int8-Export. Der PyTorch-Nachweis
belegt weder dessen GPU-Ausführung noch die Exportkompatibilität. Verglichen
werden deshalb: bisheriger TensorFlow-Build mit bisheriger Runtime, derselbe
Build mit neuem nativem SDK und das verfügbare AMD-TensorFlow-Wheel mit dem
neuen SDK. Features, Profil, Batch und Eingabepipeline bleiben gleich. Die
geladenen Bibliotheken müssen zum ausgewählten Bestand gehören.

Diese Prüfung läuft noch. Bis zum vollständigen Trainings-/Exportnachweis
bleibt die bisherige Workstation-Installation aktiv.

## Umzug nach `/opt/rocm` vorbereiten

Den neuen Bestand versioniert neben der laufenden Installation ablegen.
Bei voller Systempartition eignet sich ein dauerhaft eingebundener
Datenträger; `/opt/rocm` kann später auf den geprüften Bestand zeigen. Für
den RX-6700-XT-Vergleich ist
`therock-dist-linux-gfx103X-all-10.1.0.tar.gz` separat entpackt. Prüfsumme,
Umfang und Status bleiben in einem lokalen Vorbereitungsmanifest.

1. Verbraucher erfassen: Trainer, Teacher, LLM, TTS, Dienste, Shell,
   Linker-Konfiguration und lokale Wheel-Verweise.
2. Training, Checkpoints, int8-Export und Inferenz prüfen. LLM/TTS brauchen
   eigene Funktionstests. Der Teacher-Benchmark ersetzt sie nicht.
3. Alten Bestand als Rückweg erhalten; bei ruhiger GPU umschalten. Prozesse
   anschließend neu starten und ihre Funktion sowie Bibliothekspfade prüfen.
4. Veraltete Runtime-Verzeichnisse und Build-/Reparatur-Helfer gezielt
   entfernen; nötige Framework-Wheels separat aufbewahren. Aufnahmen,
   MMAP-Korpora, Git-Historien und Modelle bleiben Datenbestände.

Die Vorbereitung aktiviert oder entfernt keine Systeminstallation. Der
Rechnerplan muss tatsächliche Verbraucher, Umschaltung und Rückweg nennen.
