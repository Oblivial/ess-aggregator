# ess-aggregator

Aggregiert Individualdaten (Micro-Data) des **European Social Survey (ESS)**
zu Länder-Jahres-Einheiten (Country-Year), Ländern und gepoolten
europaweiten Kennzahlen, damit sie mit makroökonomischen Aggregatdaten
(Macro-Data) für Multilevel-Analysen (Hierarchische Lineare Modelle /
Panel-Regressionen) zusammengeführt werden können.

Der Datenzugriff erfolgt über die Bibliothek
[`py-ess`](https://github.com/Oblivial/py-ess), die ESS-Datensätze bedarfsgerecht
("on demand") direkt über die offizielle ESS-API lädt und mit dem ESS-Codebook
(Variablen-Labels, Wertelabels, Rundenzugehörigkeit) anreichert.

## Inhalt

- [Installation](#installation)
- [Verwendung (CLI)](#verwendung-cli)
- [Output-Format](#output-format)
- [Geschäftslogik / Berechnungen](#geschäftslogik--berechnungen)
  - [Gewichtung](#gewichtung)
  - [Beobachtungsanzahl & Mindestfallzahl](#beobachtungsanzahl--mindestfallzahl)
  - [Lagemaße: Mittelwert, Median, Modus](#lagemaße-mittelwert-median-modus)
  - [Streuung: Standardabweichung & IQR](#streuung-standardabweichung--iqr)
  - [Ungleichheit: Gini, Perzentile, Quantilsverhältnisse, Palma-Ratio](#ungleichheit-gini-perzentile-quantilsverhältnisse-palma-ratio)
  - [Mundlak-/Zero-Center-Transformation](#mundlak-zero-center-transformation)
  - [Jahres- und Länderzuordnung](#jahres--und-länderzuordnung)
- [Fehlerbehandlung & Logging](#fehlerbehandlung--logging)
- [Tests](#tests)
- [Projektstruktur](#projektstruktur)

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Dies installiert `py-ess` direkt von GitHub (siehe `pyproject.toml` /
`requirements.txt`). Für reine Entwicklungs-/Testzwecke (ohne Netzwerkzugriff
auf die ESS-API) genügt:

```powershell
pip install pandas numpy pytest
pip install -e . --no-deps
```

Die Test-Suite verwendet einen simulierten ("faked") ESS-Client (siehe
`tests/conftest.py`) und benötigt daher **kein** echtes `py-ess`/keine
Netzwerkverbindung.

## Verwendung (CLI)

```powershell
python -m ess_aggregator stflife happy --output ergebnisse.csv --log-file lauf.log
```

Wichtigste Argumente:

| Argument              | Beschreibung                                                                                          | Default                |
|------------------------|--------------------------------------------------------------------------------------------------------|-------------------------|
| `variables` (positional) | Eine oder mehrere ESS-Variablennamen, die aggregiert werden sollen (z. B. `stflife happy netusoft`).  | *erforderlich*          |
| `--rounds`             | Einschränkung auf bestimmte ESS-Runden (z. B. `ESS9 ESS10 ESS11`).                                     | alle Runden der Variable |
| `--countries`          | Einschränkung auf bestimmte Länder, per ISO-Code oder Name (z. B. `DE FR` oder `Germany France`).       | alle Länder             |
| `--min-n`              | Mindest-effektivfallzahl je Land-Jahr-Einheit (siehe unten).                                            | `300`                   |
| `--output`, `-o`       | Pfad der Ausgabe-CSV.                                                                                   | `ess_aggregated.csv`    |
| `--log-file`           | Pfad der Log-Datei.                                                                                     | `ess_aggregator.log`    |
| `--no-recode-missing`  | Deaktiviert das serverseitige Umkodieren von Missing-Values (ESS-API `recodeMissingValues`).            | aktiviert               |
| `--verbose`, `-v`      | DEBUG-Log-Level auch auf der Konsole (die Log-Datei ist immer DEBUG-Level).                              | `INFO`                  |

`py-ess` kennt zu jeder Variable automatisch, in welchen ESS-Runden sie erhoben
wurde (`Variable.rounds`), daher muss die Runde **nicht** manuell nachgeschlagen
werden - `--rounds` dient nur der optionalen Einschränkung.

## Output-Format

Für jede angeforderte Variable werden vier Arten von Aggregationseinheiten
gebildet (Beispiel für die Variable `stflife`):

| `unit`         | `unit_type`     | Bedeutung                                                        |
|----------------|-----------------|-------------------------------------------------------------------|
| `Germany2018`  | `country_year`  | Aggregat für Deutschland im Jahr 2018                             |
| `GermanyAll`   | `country_all`   | Aggregat für Deutschland über alle verfügbaren Jahre gepoolt      |
| `All2018`      | `year_all`      | Aggregat über alle Länder im Jahr 2018 gepoolt                    |
| `AllAll`       | `grand_all`     | Aggregat über alle Länder und Jahre gepoolt                       |

Jede Zeile der Ausgabe-CSV enthält folgende Spalten:

| Spalte                      | Beschreibung                                                                                     |
|-------------------------------|---------------------------------------------------------------------------------------------------|
| `unit`                       | Eindeutiger Bezeichner der Aggregationseinheit (siehe Tabelle oben).                              |
| `variable`                   | Name der aggregierten ESS-Variable.                                                                |
| `country`, `year`            | Land (Klartext-Name) bzw. Jahr der Einheit; `"All"`, wenn gepoolt.                                 |
| `unit_type`                  | `country_year` / `country_all` / `year_all` / `grand_all`.                                         |
| `n_raw`                      | Ungewichtete Anzahl gültiger (nicht-fehlender) Beobachtungen.                                      |
| `n_effective`                | Effektive Fallzahl nach Kish (siehe unten).                                                        |
| `reliable`                   | `True`, wenn `n_raw` **und** `n_effective` ≥ `--min-n`.                                            |
| `mean`, `median`, `mode`     | Gewichteter Mittelwert, Median, Modus.                                                              |
| `std`, `iqr`                 | Gewichtete Standardabweichung bzw. Interquartilsabstand (P75 − P25).                                |
| `p10`, `p40`, `p50`, `p90`   | Gewichtete Perzentile (P40 wird nur für die Palma-Ratio benötigt, aber zur Transparenz mit ausgegeben). |
| `p90_p10_ratio`              | Quantilsverhältnis P90/P10.                                                                        |
| `palma_ratio`                | Anteil der obersten 10 % geteilt durch Anteil der untersten 40 % (siehe unten).                    |
| `gini`                       | Gewichteter Gini-Koeffizient (0 = Gleichheit, 1 = maximale Ungleichheit).                          |
| `country_mean_over_years`    | Mundlak-Between-Komponente (nur für `country_year`-Zeilen; sonst `NaN`).                            |
| `within_country_deviation`   | Mundlak-Within-Komponente (nur für `country_year`-Zeilen; sonst `NaN`).                             |
| `warnings`                   | Semikolon-getrennte Warnungen zu dieser Zeile (z. B. niedrige Fallzahl).                            |

## Geschäftslogik / Berechnungen

### Gewichtung

Es wird standardmäßig `anweight` (Analysis Weight) verwendet - das von ESS
bereits fertig kombinierte Gewicht aus Design-Gewicht
(`dweight`/`pspwght`, korrigiert für ungleiche Ziehungswahrscheinlichkeiten)
**und** Populationsgrößen-Gewicht (`pweight`). Ist `anweight` in einer Runde
nicht vorhanden, wird es aus den Komponenten `pspwght`/`dweight * pweight`
rekonstruiert; ist auch das nicht möglich, wird auf ungewichtete Analyse
(`weight = 1.0`) zurückgefallen - **mit lautem Log-Warning**, da dies die
Stichprobenverzerrung nicht mehr korrigiert.

Warum `anweight` sowohl für Länder- als auch für gepoolte Analysen sicher ist:

- `pweight` ist **innerhalb** eines Land-Runde-Blocks für alle Befragten
  konstant (es skaliert nur die Grundgesamtheit relativ zur Stichprobengröße).
  Da gewichteter Mittelwert, Median und alle Perzentile **invariant gegenüber
  einer konstanten Skalierung aller Gewichte** sind (siehe
  `tests/test_aggregation.py::TestWeightedMean::test_scale_invariance_to_constant_weight_multiplier`),
  liefert `anweight` für `country_year`-/`country_all`-Einheiten (ein einzelnes
  Land) **exakt dieselben** gewichteten Statistiken wie `pspwght`/`dweight`
  allein.
- Für gepoolte Einheiten (`year_all`, `grand_all`, also mehrere Länder
  zusammen) sorgt `pweight` hingegen korrekt dafür, dass größere Länder
  proportional zu ihrer Bevölkerung stärker gewichtet werden - das ist
  genau der Zweck von `anweight` bei länderübergreifenden Vergleichen.

Vor der Berechnung der Verteilungskennzahlen werden Beobachtungen mit
fehlendem Wert (`NaN`, durch die ESS-API bereits als System-Missing kodiert)
oder nicht-positivem Gewicht entfernt (`ess_aggregator/aggregation.py::_clean`).

### Beobachtungsanzahl & Mindestfallzahl

Pro Land-Jahr-Einheit wird zusätzlich zur rohen Fallzahl (`n_raw`) die
**effektive Fallzahl** nach der Kish-Approximation berechnet:

```
n_eff = (Σ w_i)² / Σ (w_i²)
```

`n_eff` misst, wie viele *gleichgewichtete* Befragte denselben
Informationsgehalt hätten wie die tatsächlich ungleich gewichtete Stichprobe.
Sie ist ebenfalls invariant gegenüber einer konstanten Gewichts-Skalierung.

Liegt `n_raw` **oder** `n_eff` unter dem Schwellenwert `--min-n`
(Default `300`, gemäß Vorgabe "N < 300–500"), werden alle
verteilungsbasierten Kennzahlen (Median, P10/P40/P50/P90, IQR, Gini,
Quantilsverhältnisse, Palma-Ratio) als `NaN` ausgegeben und die Zeile mit
`reliable = False` sowie einer entsprechenden Warnung markiert - diese
Kennzahlen reagieren bei kleinen Stichproben besonders empfindlich auf
Ausreißer. Mittelwert, Modus und Standardabweichung werden weiterhin
ausgegeben, da sie deutlich robuster gegenüber kleinen `n` sind, aber ebenso
mit `reliable = False` gekennzeichnet.

### Lagemaße: Mittelwert, Median, Modus

- **Mittelwert**: `Σ(w·x) / Σw`.
- **Median**: gewichtetes Perzentil bei `q = 0.5` (siehe Perzentil-Methode
  unten).
- **Modus**: der Wert mit der größten Gewichtssumme (`Σw` je Ausprägung, dann
  Maximum). Für kategoriale/diskrete Variablen (z. B. Zustimmungsskalen) ist
  das sinnvoll interpretierbar; bei quasi-kontinuierlichen Variablen liefert
  es lediglich den am häufigsten/stärksten gewichteten Einzelwert, keine
  dichte-geschätzte Modalstelle - diese Einschränkung ist zu beachten.

### Streuung: Standardabweichung & IQR

- **Gewichtete Standardabweichung**: Quadratwurzel der gewichteten
  (Populations-)Varianz `Σw·(x − x̄)² / Σw` um den gewichteten Mittelwert.
- **IQR (Interquartilsabstand)**: `P75 − P25`, jeweils gewichtete Perzentile.

Gewichtete Perzentile werden nach der **Mittelpunkt-/Hazen-Methode**
berechnet (wie z. B. `Hmisc::wtd.quantile` in R): Werte werden aufsteigend
sortiert, jeder Beobachtung `i` wird die kumulierte Gewichtsanteil-Mitte

```
cw_i = (cumsum(w)_i − 0.5·w_i) / Σw
```

zugeordnet, und das gewünschte Perzentil wird linear zwischen den
umliegenden `cw_i` interpoliert. Bei Gleichgewichtung entspricht dies dem
üblichen unwewichteten Perzentil.

### Ungleichheit: Gini, Perzentile, Quantilsverhältnisse, Palma-Ratio

- **Gini-Koeffizient** (gewichtet): über die diskrete
  Lorenzkurven-/Trapez-Formel für gruppierte Daten - Werte aufsteigend
  sortiert, kumulierter Gewichtsanteil `w_share_i` und kumulierter
  Werteanteil (Lorenzkurven-Ordinate) `L_i`, dann

  ```
  G = 1 − Σ_i w_share_i · (L_{i-1} + L_i)
  ```

  Dies entspricht der doppelten Fläche zwischen der Lorenzkurve und der
  Gleichverteilungslinie. Der Gini ist nur für nicht-negative Größen
  definiert (z. B. Einkommen, Zufriedenheitsskalen ab 0) - bei negativen
  Werten wird `NaN` zurückgegeben. Die Implementierung wurde in
  `tests/test_aggregation.py::TestGini` gegen eine O(n²)-Referenzimplementierung
  (Brute-Force-Paarvergleich) verifiziert.
- **P10 / P50 / P90**: gewichtete Perzentile an den Rändern und der Mitte der
  Verteilung, um Effekte an den Rändern separat zu testen.
- **P90/P10-Quantilsverhältnis**: `P90 / P10` (bei `P10 = 0` `NaN`, um
  Division durch Null zu vermeiden).
- **Palma-Ratio**: Anteil der Gesamtsumme (gewichtet), den die obersten 10 %
  (Werte ≥ P90) halten, geteilt durch den Anteil der untersten 40 % (Werte
  ≤ P40):

  ```
  Palma = ( Σ w·x  für x ≥ P90 ) / (Σ w·x insgesamt)
          -------------------------------------------
          ( Σ w·x  für x ≤ P40 ) / (Σ w·x insgesamt)
  ```

  Die Palma-Ratio ist eine gängige Alternative zum Gini, die sich stärker auf
  die Verteilungsränder konzentriert und weniger empfindlich auf Rauschen in
  der Verteilungsmitte reagiert.

### Mundlak-/Zero-Center-Transformation

Um unbeobachtete länderspezifische Heterogenität (Kultur, Geschichte, ...)
von echten Zeitveränderungen zu trennen, wird für jede
`country_year`-Zeile die klassische Mundlak-(1978)-Zerlegung des
Aggregat-Mittelwerts (`mean`) berechnet:

- **`country_mean_over_years`** (Between-Country-Effekt): das einfache
  arithmetische Mittel des jährlichen `mean`-Werts eines Landes über alle
  Jahre, in denen Daten vorliegen:

  ```
  X̄_c = (1/T_c) · Σ_t X_ct
  ```

  Hinweis: Dies ist ein **einfacher Durchschnitt der Jahres-Kennzahlen**, nicht
  ein erneut über alle Rohdaten gepoolter (nach N gewichteter) Mittelwert -
  das entspricht der Standard-Mundlak-Praxis auf Panel-Ebene und ist bewusst
  von `GermanyAll` (der direkt aus allen Rohbeobachtungen gepoolten Kennzahl)
  zu unterscheiden.
- **`within_country_deviation`** (Within-Country-Effekt): Abweichung des
  jeweiligen Jahres vom eigenen Ländermittelwert:

  ```
  X_ct − X̄_c
  ```

  Per Konstruktion summieren sich die `within_country_deviation`-Werte eines
  Landes über alle Jahre zu 0 (siehe
  `tests/test_pipeline.py::TestMundlakDecomposition`).

Beide Spalten sind für die gepoolten Einheiten (`country_all`, `year_all`,
`grand_all`) nicht definiert (`NaN`), da das Between/Within-Konzept
begrifflich nur auf der Land-Jahr-Ebene sinnvoll ist.

### Jahres- und Länderzuordnung

- **Land**: aus der ESS-Variable `cntry` (ISO-3166-1-alpha-2-Code), über das
  Codebook in den Klartextnamen dekodiert (z. B. `"DE"` → `"Germany"`).
- **Jahr**: aus der präzisesten verfügbaren Interview-Jahr-Variable, in dieser
  Prioritätsreihenfolge: `inwyys` (Beginn des Interviews, Jahr) →
  `inwyr` (ältere Runden, einzelnes Interviewjahr) → `inwyye` (Ende des
  Interviews, Jahr). Das tatsächliche Interviewjahr pro Befragten ist
  präziser als die nominelle Runden-Jahreszahl, da die Feldarbeit einzelner
  Länder z. T. über einen Jahreswechsel hinausreicht.

## Fehlerbehandlung & Logging

- Jeder Verarbeitungsschritt wird sowohl auf der Konsole als auch in der
  `--log-file` protokolliert (Zeitstempel, Log-Level, Modul, Nachricht).
- **Fehler pro Runde** (z. B. Netzwerkfehler, fehlende Spalte) führen nicht
  zum Abbruch des gesamten Laufs: die betroffene Runde wird übersprungen und
  im Log als Fehler vermerkt, die übrigen Runden werden weiterverarbeitet.
- **Fehler pro Variable** (z. B. unbekannter Variablenname, keine Runde mehr
  verfügbar) führen dazu, dass diese Variable übersprungen wird, sofern
  mindestens eine andere Variable erfolgreich geladen werden kann.
- Schlagen **alle** angeforderten Variablen fehl, oder tritt ein sonstiger
  unerwarteter Fehler auf, terminiert das Programm mit einem Exit-Code
  ungleich 0 und einer vollständigen Fehlermeldung/Traceback im Log.
- Exit-Codes: `0` = Erfolg, `1` = Fataler Verarbeitungsfehler / keine Zeilen
  erzeugt, `2` = `py-ess` ist nicht installiert.

## Tests

```powershell
pytest -q
```

Die Test-Suite (siehe `tests/`) deckt ab:

- `test_aggregation.py`: alle statistischen Bausteine (gewichteter Mittelwert/
  Median/Modus/Std, Gini inkl. Brute-Force-Gegenprobe, Perzentile,
  Quantilsverhältnisse, Palma-Ratio, effektive Fallzahl, Mindestfallzahl-Logik).
- `test_data_loader.py`: Runden-Auflösung, Gewichts-/Jahres-Spalten-Fallbacks,
  Länder-Dekodierung, robuste Fehlerbehandlung bei kaputten Runden.
- `test_pipeline.py`: Aufbau der Aggregationseinheiten (`Germany2018`,
  `GermanyAll`, `All2018`, `AllAll`), Mundlak-Zerlegung, Multi-Variablen-Lauf.
- `test_cli.py`: Argument-Parsing und End-to-End-Lauf mit simuliertem
  ESS-Client (kein Netzwerkzugriff nötig).

## Projektstruktur

```
ess-aggregator/
├── src/ess_aggregator/
│   ├── aggregation.py   # reine Statistik-Funktionen (gewichtet)
│   ├── cli.py           # Kommandozeilen-Schnittstelle
│   ├── config.py        # zentrale Konstanten (Gewichte, Jahres-Spalten, Mindest-N, ...)
│   ├── data_loader.py    # py-ess-Anbindung: lädt Variablen über Runden hinweg
│   ├── exceptions.py     # eigene Fehlerklassen
│   ├── logging_utils.py  # Logging-Konfiguration
│   ├── output.py         # CSV-Schreiber
│   └── pipeline.py       # Orchestrierung: Gruppierung, Aggregation, Mundlak
├── tests/                # pytest-Suite (inkl. fake-ESS-Client in conftest.py)
├── pyproject.toml
├── requirements.txt
└── README.md
```
