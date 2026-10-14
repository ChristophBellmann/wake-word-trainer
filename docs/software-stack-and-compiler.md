# Softwarestack, Compiler und Optimierung

Diese Seite erklärt die technischen Entscheidungen dieser Anwendung. Stand:
14.10.2026. Die Workstation hat eine RX 6700 XT mit 12 GiB VRAM, zwölf
CPU-Kerne mit 24 logischen CPUs und etwa 31 GiB RAM. Die Ergebnisse sind
Messungen an diesem Rechner, keine allgemeinen Leistungsversprechen.

## Welche Software welche Aufgabe übernimmt

| Bestandteil | Aufgabe hier | Ausführung |
| --- | --- | --- |
| Python und NumPy | Pipeline, Datenverwaltung, Audiofenster und Prüfreports | CPU |
| `pymicro-features` / Microfrontend | Student-Features passend zum Mikrocontroller: 40 Bänder, ein Frame alle 10 ms | CPU |
| TensorFlow und Keras | Student-Modell, gewichteter Loss, Adam, Checkpoints und Konvertierung | Training auf GPU; Export und Streaming-Prüfung auf CPU |
| microWakeWord | Modellarchitektur und Streaming-Modell für ESPHome | TensorFlow beim Training; TFLite Micro auf dem Satelliten |
| PyTorch | Optionaler Log-Mel-Teacher, Bewertung und eigenes Teacher-Training mit AdamW | GPU, wenn angefordert und verfügbar |
| Piper und ONNX Runtime | Synthetische positive und ähnliche negative Sprache | CPU |
| faster-whisper | Unabhängige STT-Prüfung von Labels und Fehlaktivierungen | Hier CPU; eigener optionaler Installationsumfang |
| ROCm/HIP | GPU-Laufzeit und Verbindung der Frameworks zur AMD-GPU | GPU-Runtime |
| MIOpen, rocBLAS und RCCL | Faltungs-/Matrixoperationen und Framework-Abhängigkeiten | GPU-Bibliotheken; jeweils passend zum Stack |
| Clang, `hipcc`, LLD und Bazel | Framework-Wheel aus C++-/HIP-Quellen bauen | Kompilierung auf CPU; erzeugte GPU-Kernels laufen später auf GPU |

Ein TensorFlow-Neubau, Audioaufbereitung oder int8-Vergleich beschäftigt
deshalb die CPU. Eine ruhende GPU in diesen Phasen erklärt sich durch die
Aufgabe. Im Training muss dagegen eine abgeschlossene GPU-Operation den
gewählten Pfad nachweisen. `training.device: gpu` bricht bei einem fehlenden
GPU-Backend ab; siehe [GPU-Training](gpu-training-and-benchmarks.md).

Der Teacher verwendet eigene Log-Mel-Features und die Architekturen
`legacy_v2` oder `temporal_v1`. Der Student verwendet das Microfrontend.
Ihre Features sind nicht austauschbar. Der Teacher liefert zusätzliche
Prüfvorschläge; er ersetzt weder STT noch bestätigte Labels. Der öffentliche
Student trainiert weiterhin mit den bestehenden Labels, ohne automatisch
einen Soft-Label-Distillationsloss zu übernehmen. Die Prüfschritte stehen
unter [Daten prüfen](data-audit.md).

## Compiler für Host und GPU

Im alten Stack wählte `~/.bashrc` bewusst `/opt/rocm/llvm/bin` für Clang.
`llvm` verweist dort auf `lib/llvm`, `clang` auf AMD Clang 22. Der neue
native SDK-Bestand enthält dieselbe Verzeichnisstruktur mit AMD Clang 24.

Der aktuelle TensorFlow-Neubau trennt die Aufgaben:

| Werkzeug | Tatsächliche Auswahl | Zweck |
| --- | --- | --- |
| Hostcompiler | Ubuntu Clang 18.1.3 | C++-Code für die Workstation |
| GPU-Compiler | Neues SDK, `hipcc` mit AMD Clang 24 | HIP-Kernels für `gfx1031` |
| Host-Linker | LLD 18.1.3 | Kompilierte Hostobjekte und Bibliotheken zusammenführen |
| Buildsystem | Bazel 7.4.1 | Abhängigkeiten, Aktionen und Wiederverwendung im Cache |

Der Hostcompiler wird mit einem absoluten Pfad gesetzt. Der erzeugte
ROCm-Crosstool-Wrapper verwendet für GPU-Code `hipcc` aus dem ausgewählten
SDK. Das vermeidet eine unbeabsichtigte Auswahl aus einer alten Shell.
Compiler-Version und reale Datei erfassen:

```bash
command -v clang
readlink -f "$(command -v clang)"
clang --version
command -v hipcc
hipcc --version
```

Zusätzlich `~/.bashrc`, `/etc/profile.d/` und die Dienstumgebung prüfen.
Die Workstation hatte neben dem kanonischen `.bashrc`-Pfad noch Login-
Verweise auf das fehlende `/opt/rocm-7.1.0`. Deren Ersatz ist vorbereitet.
Die beabsichtigte Clang-Auswahl wird erhalten; siehe [ROCm-Umzug](rocm-update.md).

Ein direkt mit dem neuen `hipcc` gebauter `gfx1031`-Kernel bestand die Prüfung
aller 256 GPU-Ergebnisse. Keine Bibliothek aus der alten `/opt/rocm`-
Installation war geladen. Dieser Compiler-Test ist bestanden; der vollständige
TensorFlow-Wheel-Test bleibt eine eigene Prüfung.

## Welche Build-Optimierungen gesetzt sind

Die Quellen sind festgelegt: AMD `ROCm/tensorflow-upstream`, Branch
`r2.20-rocm-enhanced`, Commit `2efcf3ba0da8b453902ac723086127d43c9af65c`.
Python-Version und GPU-Ziel stehen zusätzlich im Build-Bericht.

| Einstellung | Bedeutung in diesem Build |
| --- | --- |
| `--config=opt`, `-O3` | Optimierter Hostcode; die verwendete Konfiguration setzt `-O3` auch für Hostwerkzeuge. |
| `TF_ROCM_AMDGPU_TARGETS=gfx1031` | GPU-Kernels für die vorhandene Karte. Keine vorgetäuschte andere GPU-Architektur. |
| `--jobs=24`, CPU-Ressourcen 24 | Parallele unabhängige Build-Aktionen auf allen logischen CPUs. |
| RAM-Scheduling-Limit 18 GB | Begrenzt die vom Buildsystem eingeplante Arbeit; kein harter Prozessspeicherdeckel. |
| `-fuse-ld=lld` | Explizite Auswahl des vorhandenen Host-Linkers LLD. |
| XLA-JIT ausgeschaltet | Der Trainer verwendet TensorFlow-Graph-Ausführung ohne XLA-JIT. GPU-Training bleibt aktiviert. |
| `-Wl,-Bsymbolic-functions` | Bindet interne Funktionsaufrufe einer Shared Library an ihre eigenen Definitionen. Soll die beobachtete LLVM-/MLIR-Kollision vermeiden; der fertige Wheel-Test muss dies bestätigen. |
| SDK-RPATH und `rocm_path_type=hermetic` | Bibliotheken aus dem vorgesehenen SDK wählen, statt den Standardpfad der alten Runtime zu übernehmen. |

`-O3` wird hier nicht durch zusätzliche `-ffast-math`-Flags ergänzt.
Compileroptimierung, GPU-Ziel und numerisches Verhalten sind getrennte
Entscheidungen. Die Optionen beschreibt die
[Clang-Dokumentation](https://clang.llvm.org/docs/CommandGuide/clang.html);
die Bedeutung von `-Bsymbolic-functions` steht in der
[Linker-Dokumentation](https://sourceware.org/binutils/docs/ld/Options.html).
Das Abschalten von XLA-JIT entfernt nicht sämtliche LLVM-/MLIR-Bausteine
aus dem Framework-Build: Auch Modellkonvertierung und weitere Werkzeuge
benötigen Compiler-Infrastruktur.

Nach Freigabe aller 24 CPUs wurden über fünf Sekunden **99,3 % CPU-Auslastung**
gemessen, entsprechend etwa 23,8 beschäftigten logischen Kernen. Es liefen
25 Clang-Prozesse; verfügbarer RAM: 18,6 GiB. Unterprozesse können die Anzahl
der Build-Aktionen übersteigen. Abhängigkeiten und serielle Linkerphasen
begrenzen die Auslastung in anderen Phasen. Der Ressourcenwächter beendet
den Build regulär, wenn weniger als 30 GiB Plattenreserve oder 2 GiB RAM
verfügbar sind. Caches und temporäre Dateien liegen auf der NVMe.

## Generatoren und Bibliothekssuchpfade

TensorFlow erzeugt während des Builds C++- und Python-Schnittstellen mit
eigenen nativen Werkzeugen. Auch diese Werkzeuge benötigen die richtigen
ROCm-Bibliotheken. Bazels `_solib`-Verknüpfungen machten den relativen
LLVM-Suchpfad von COMGR unbrauchbar. Ein zusätzlicher RPATH am ausführbaren
Programm allein löste den Fehler nicht.

Die beiden Generatorbefehle erhalten deshalb gezielt `LD_LIBRARY_PATH` mit
`$ROCM_PATH/lib`, `$ROCM_PATH/lib/llvm/lib` und
`$ROCM_PATH/lib/rocm_sysdeps/lib`. Ein globales neues Action-Environment
würde auch bereits kompilierte Dateien im Cache entwerten. Die begrenzte
Korrektur ist mit beiden Bazel-Zielen geprüft:
`//tensorflow/cc:lookup_ops_genrule` und
`//tensorflow/python/user_ops:user_ops_pygenrule`.

Für die Fehlerdiagnose unterscheiden wir fehlende Bibliotheken,
Versions-/Symbolkollisionen und GPU-Ausführungsfehler. Das angebotene
AMD-TensorFlow-Wheel für ROCm 7.2.1 scheiterte mit SDK 10.1 an einer
LLVM-/MLIR-Kollision beziehungsweise am Prozessabschluss. Ein Paketimport
allein ist deshalb kein vollständiger Funktionstest.

## Optimierung des Trainings ist eine weitere Ebene

Der [optimierte Student-Kern](https://github.com/ChristophBellmann/wake-word-trainer/blob/50434d4/src/wake_word_trainer/optimized_tf.py)
verwendet `tf.function(reduce_retracing=True, jit_compile=False)`.
TensorFlow führt den Trainingsschritt als Graph aus und reduziert wiederholte
Python-Aufrufe. Das ist von einer XLA-JIT-Kompilierung zu unterscheiden;
siehe [TensorFlow zu `tf.function`](https://www.tensorflow.org/guide/function).

Gewichtete binäre Kreuzentropie, Adam und Checkpoint-Auswahl bleiben erhalten.
Sample-Gewichte und Klassen-Gewichte werden zusammengeführt; die bestehende
Loss-Normalisierung bleibt bestehen. Zeit-/Frequenzmasken entstehen für jeden
Batch neu auf GPU nach dem statischen Input-Cache.

Die [Eingabepipeline](https://github.com/ChristophBellmann/wake-word-trainer/blob/50434d4/src/wake_word_trainer/input_pipeline.py)
überlappt Zuführung und Rechenarbeit über Prefetch beziehungsweise TFRecord.
Stamps binden Dateien, Metadaten, Formen, Gewichte und Seed. Der Standardpool
hat 65.536 gezogene Beispiele; ein Korpus mit einer Million Einträgen liegt
damit nicht automatisch vollständig in jedem Trainingslauf. Ein größerer
Pool benötigt einen eigenen Qualitäts- und Laufzeitvergleich.
Die Grundprinzipien beschreibt die
[TensorFlow-Anleitung zur Eingabepipeline](https://www.tensorflow.org/guide/data_performance).

Vollständige Validierungsdaten werden nur innerhalb des konfigurierten
RAM-Limits gecacht. Audio-Augmentation und Microfrontend-Bau laufen in
begrenzten CPU-Prozessen mit `spawn` und ausgeblendeter GPU. Die gemeinsame
GPU-Dateisperre gilt für komplette Trainings-/Benchmark-Läufe, einschließlich
CPU-Phasen zwischen GPU-Arbeiten.

Auch Datentypen gehören zur Modellkompatibilität. Ambient-MMAPs mit
`uint16`-Microfrontend-Werten müssen ihren Typ bis zum Inferenzmodell behalten,
damit dessen Skalierung mit `0.0390625` greift. Ein vorheriger Float-Cast hatte
dies verhindert und wurde korrigiert. Für quantisierte Modelle werden die
8-Bit-Wahrscheinlichkeiten im gleitenden Mittel mit ganzzahligen Summen
ausgewertet. So entstehen durch kumulierten Float-Fehler keine vermeintlichen
Treffer oberhalb von 1. Alte 20-ms-Features bleiben vom 10-ms-Student-Pfad
ausgeschlossen; siehe [Negativkorpora](negative-corpora.md).

Größere Batches erhöhen häufig Beispiele/s, können aber Schritte/s und
Modellqualität anders verändern. Sie müssen deshalb am gleichen Holdout
verglichen werden. Die [Benchmark-Anleitung](gpu-training-and-benchmarks.md)
enthält die CLI-Aufrufe, Parameter und bisherigen Batch-Messungen.

## Welche Verbesserungen bereits gemessen sind

| Vergleich auf derselben RX 6700 XT | Ergebnis | Aussagegrenze |
| --- | --- | --- |
| Student: bisheriger Kern gegen optimierten Kern, Batch 256/Prefetch | Etwa 32,3 → 65,1 Schritte/s | Technische Kernoptimierung; keine belegte bessere Erkennung. |
| Derselbe bisherige TensorFlow-Build mit alter/neuer nativer Runtime | Median etwa 49,7 → 62,9 Schritte/s, zweimal 50 Aufwärm-/250 Messschritte | Rund 27 % mehr Durchsatz; angepasste alte Wheels müssen trotzdem ersetzt werden. |
| Teacher: bisheriger Eigenbau gegen AMD Torch 2.12/SDK 10.1, Batch 256 | Median etwa 9.826 → 14.674 Beispiele/s, zwei Wiederholungen | Rund 49 %; Framework und Runtime ändern sich gemeinsam. |
| Neuer Clang-24-/HIP-Kernel | 256/256 GPU-Werte korrekt | Compiler-Nachweis; kein TensorFlow-Training. |
| C++-/Python-Op-Generatoren nach Loader-Korrektur | Beide Bazel-Ziele erfolgreich | Build-Reparatur; neues vollständiges Wheel noch nicht freigegeben. |

Die Messreihen haben feste Eingaben, getrennte Aufwärmzeit und im zweiten
Durchgang umgekehrte Reihenfolge. Geladene Bibliotheken und tatsächlich
abgeschlossene GPU-Arbeit sind geprüft. Details und Einschränkungen stehen
im [ROCm-Vergleich](rocm-update.md). Die FAPH-/Recall-Vergleiche der Korpora
stehen separat unter [Negativkorpora](negative-corpora.md).

## Wann ein neues Wheel in den Zielbestand darf

Der neue SDK-Bestand ist vorbereitet. AMD-PyTorch samt MIOpen und Teacher-CLI
ist separat geprüft; das neue TensorFlow-Wheel wird noch gebaut. Die
anschließende automatische Prüfung verwendet ausschließlich eine neue
Python-Umgebung und erfasst:

1. Wheel-Hash, Quellstand, Compiler und SDK-Version.
2. Frischen CPU-Import mit sauberem Prozessabschluss.
3. Echte GPU-Operation und Bibliothekspfade ohne alte Runtime.
4. 1.000 Student-Schritte, Checkpoints und Wiederherstellung beim Export.
5. int8-Konvertierung und Streaming-Inferenz auf einer Prüfaufnahme.
6. Zwei Durchsatzmessungen auf unveränderten Features.

Die alte Runtime und ihre angepassten Wheels bleiben bis zu einem gemeinsam
geprüften Austauschbestand erhalten. `tensorflow-rocm` und `tensorflow`
sind für pip unterschiedliche Distributionsnamen, obwohl beide das Modul
`tensorflow` bereitstellen können. Die [AMD-Installation](installation.md)
verhindert einen zusätzlichen Framework-Download. Separate Student-/Teacher-
Umgebungen sind möglich; alte globale Loader-Werte werden nicht übernommen.

Ein technisch gültiges Wheel oder ein schnelleres Training gibt noch kein
Satellitenmodell frei. Das Modell braucht den Vergleich auf identischen
Prüfdaten, ein eingehaltenes Fehlaktivierungsbudget und einen bestandenen
Parity-Test. Aufnahmen, Transkripte, Modelle und private Laufberichte bleiben
außerhalb des öffentlichen GitBooks.
