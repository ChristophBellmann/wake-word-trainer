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
teacher-rocm10/bin/python -m pip install 'numpy==1.26.4' \
  'wake-word-trainer @ git+https://github.com/ChristophBellmann/wake-word-trainer'
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

Mit demselben bisherigen TensorFlow-Build wurde der neue native SDK-Bestand
geprüft: zweimal 50 Aufwärm- und 250 Messschritte, Batch 256, gleiche
TFRecord-Features, umgekehrte Reihenfolge. Der Median stieg von etwa
**49,7 auf 62,9 Schritte/s**, rund **27 %**. MIOpen 3.6.1 und die weiteren
GPU-Bibliotheken wurden aus dem neuen Ziel geladen; die Prüfung schließt
alte `/opt/rocm`-Bibliotheken im neuen Lauf ausdrücklich aus.

Ein separater Lauf mit 1.000 GPU-Schritten bestand Checkpoints, int8-Konvertierung
und Streaming-Inferenz. Das ist ein Funktionsnachweis der neuen Runtime mit
diesem Build; es ist noch kein vollständiger Wheel-Austausch.

## Angepasste alte Wheels ebenfalls ersetzen

Die alten Eigenbau-Wheels gehören zur alten Umgebung. Der Test mit dem
bisherigen TensorFlow-Build dient als Zwischennachweis. Der Zielplan ersetzt
auch diese Pakete durch passende aktuelle Wheels, jeweils mit gemeinsamem
GPU-/Trainings-/Exporttest. Torch ist bereits durch die stabilen AMD-Pakete
ersetzt und in einer getrennten Umgebung geprüft.

Das verfügbare AMD-TensorFlow-Wheel für ROCm 7.2.1 stürzte mit dem neuen SDK
beim Import ab; ohne die alten Preloads war außerdem ein fehlerhafter
Prozessabschluss zu beobachten. Der Debugger zeigt eine Kollision von
TensorFlow-LLVM/MLIR mit dem SDK-LLVM. Auch der alte Torch-Build bestand die
Kombination mit neuer Runtime nicht. Bibliothekskopieren allein ist deshalb
kein vollständiger Wechselplan.

Für TensorFlow wird ein neues Wheel aus festgelegtem AMD-Quellstand gegen
SDK 10.1 und `gfx1031` vorbereitet. Es ersetzt erst nach eigenem Nachweis das
alte angepasste Wheel. Tatsächlich verwendete weitere GPU-Pakete und die
Interpreter der LLM-/TTS-Dienste werden getrennt erfasst. Nicht mehr genutzte
Lab-Pakete werden nicht vorsorglich in den neuen Trainer installiert.

Die aktive Installation bleibt bis zu einem gemeinsam geprüften Zielbestand
erhalten. Die native SDK-Datei und die neuen Python-SDK-Pakete sind vorbereitet;
der vollständige Paketwechsel bleibt ein eigener, noch nicht freigegebener Schritt.

## Umzug nach `/opt/rocm` vorbereiten

Den neuen Bestand versioniert neben der laufenden Installation ablegen.
Bei voller Systempartition eignet sich ein dauerhaft eingebundener
Datenträger; `/opt/rocm` kann später auf den geprüften Bestand zeigen. Für
den RX-6700-XT-Vergleich ist
`therock-dist-linux-gfx103X-all-10.1.0.tar.gz` separat entpackt. Prüfsumme,
Umfang und Status bleiben in einem lokalen Vorbereitungsmanifest.

1. Verbraucher erfassen: Trainer, Teacher, LLM, TTS, Dienste, Shell,
   Linker-Konfiguration, Compiler-Auswahl und lokale Wheel-Verweise.
2. Training, Checkpoints, int8-Export und Inferenz prüfen. LLM/TTS brauchen
   eigene Funktionstests. Der Teacher-Benchmark ersetzt sie nicht.
3. Alten Bestand als Rückweg erhalten; bei ruhiger GPU umschalten. Prozesse
   anschließend neu starten und ihre Funktion sowie Bibliothekspfade prüfen.
4. Veraltete Runtime-Verzeichnisse und Build-/Reparatur-Helfer gezielt
   entfernen; nötige Framework-Wheels separat aufbewahren. Aufnahmen,
   MMAP-Korpora, Git-Historien und Modelle bleiben Datenbestände.

Die Vorbereitung aktiviert oder entfernt keine Systeminstallation. Der
Rechnerplan muss tatsächliche Verbraucher, Umschaltung und Rückweg nennen.

Eine frühere Verknüpfung oder ein Shell-Suchpfad für ROCm-Clang kann bewusst
gesetzt worden sein. Vor dem Wechsel die tatsächliche Datei mit
`command -v clang`, `readlink -f "$(command -v clang)"` und `clang --version`
erfassen; für `hipcc` ebenso. Framework-Builds dürfen Host- und GPU-Compiler
getrennt auswählen. Beide Pfade und Versionen gehören dann in den
Build-Nachweis. Die Workstation verwendet bisher AMD Clang 22 aus
`/opt/rocm/llvm/bin`; das neue native SDK enthält AMD Clang 24. Der laufende
TensorFlow-Neubau wählt explizit Clang 18 für Hostcode und das neue SDK für
HIP. Die bestehende Shell-Auswahl bleibt während der Vorbereitung erhalten.
