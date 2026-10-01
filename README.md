# ess-aggregator

Aggregate individual-level microdata from the **European Social Survey (ESS)**
into country-year, country-wide, and pooled European statistics. The resulting
CSV or Parquet file can be joined to macroeconomic data for multilevel
analyses, hierarchical linear models, and panel regressions.

ESS data is loaded on demand through
[`py-ess`](https://github.com/Oblivial/py-ess), which uses the official ESS API
and provides codebook metadata such as variable labels and round membership.

## Contents

- [Installation](#installation)
- [Development setup](#development-setup)
- [Command-line usage](#command-line-usage)
- [Output format](#output-format)
- [Calculations and business logic](#calculations-and-business-logic)
  - [Weights](#weights)
  - [Sample size and reliability threshold](#sample-size-and-reliability-threshold)
  - [Central tendency: mean, median, and mode](#central-tendency-mean-median-and-mode)
  - [Dispersion: standard deviation and IQR](#dispersion-standard-deviation-and-iqr)
  - [Inequality: Gini, percentiles, quantile ratio, and Palma ratio](#inequality-gini-percentiles-quantile-ratio-and-palma-ratio)
  - [Mundlak / zero-centering transformation](#mundlak--zero-centering-transformation)
  - [Country and year identification](#country-and-year-identification)
- [Error handling and logging](#error-handling-and-logging)
- [Tests](#tests)
- [Project structure](#project-structure)

## Installation

Download or clone this repository, open a terminal in the project directory,
and install the package and its dependencies:

```powershell
python -m pip install .
```

Then run the command-line tool:

```powershell
ess-aggregate stflife happy --output results.csv
```

The installation includes `py-ess` `v0.1.0b3`, installed directly from GitHub
as configured in `pyproject.toml`. Data is fetched from the ESS API when the
program runs. Set your registered ESS API user ID via the `PYESS_USER_ID`
environment variable (see `.env.example`); get one at
https://ess.sikt.no/en/api.

## Development setup

### Developing py-ess and ess-aggregator together

If you're working on both `py-ess` and `ess-aggregator` at once (e.g. both
checked out as sibling directories), plain `pip install .` is a trap: it
silently pulls `py-ess` from the pinned git tag instead of using your local
`py-ess` checkout, so edits to `py-ess` appear to have no effect. Use the
provided setup script instead, which creates a shared venv and installs both
packages in *editable* mode:

```powershell
.\scripts\dev-setup.ps1
..\.venv\Scripts\Activate.ps1
```

Verify both resolved to your local checkouts (not a stale/pinned copy) with:

```powershell
pip show py-ess ess-aggregator
```

Look for `Editable project location` pointing at your checkouts. The CLI also
logs the resolved `py-ess` version and source file on every run (visible with
`--verbose` or in the log file), so drift is easy to spot later.

Copy `.env.example` to `.env` and fill in your ESS user ID; it's loaded
automatically (and is gitignored, so it never gets committed).

### Developing ess-aggregator alone

To work on `ess-aggregator` only, create a virtual environment and install
the package in editable mode together with the test dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the test suite with:

```powershell
python -m pytest
```

The tests use a fake ESS client and do not require `py-ess` or network access
at test time. To install only the local package's basic test dependencies
without installing `py-ess`, use:

```powershell
python -m pip install pandas numpy pytest
python -m pip install -e . --no-deps
```

## Command-line usage

```powershell
ess-aggregate stflife happy --output results.csv --log-file run.log
```

| Argument | Description | Default |
|---|---|---|
| `variables` (positional) | One or more ESS variable names, for example `stflife happy netusoft`. | Required |
| `--rounds` | Restrict processing to specific ESS rounds, for example `ESS9 ESS10 ESS11`. | Every round containing the variable |
| `--countries` | Restrict processing to ISO alpha-2 country codes or country names, for example `DE FR` or `Germany France`. | All countries |
| `--min-n` | Minimum effective sample size required to report distributional statistics. | `300` |
| `--output`, `-o` | Output path. `.parquet` and `.pq` suffixes select Parquet automatically; otherwise CSV is used. | `ess_aggregated.csv` |
| `--output-format` | Explicit output format: `csv` or `parquet`. | Inferred from `--output` |
| `--input-csv` | Load a local ESS CSV instead of downloading rounds from the ESS API. | Download from the API |
| `--engine` | Dataframe engine for `--input-csv`: `pandas` or `polars`. | `pandas` |
| `--log-file` | Processing and error log path. | `ess_aggregator.log` |
| `--no-recode-missing` | Do not ask the ESS API to recode designated missing values to system missing values. | Recode missing values |
| `--verbose`, `-v` | Enable DEBUG-level console logging. The log file always uses DEBUG level. | INFO |

The codebook identifies which rounds contain each variable, so round names do
not need to be looked up or supplied unless you want to restrict the analysis.

Show all options with:

```powershell
ess-aggregate --help
```

### Load a local CSV with Polars

Polars is optional. Install the extra with:

```powershell
python -m pip install ".[polars]"
```

Then pass the local ESS CSV and select the Polars engine:

```powershell
ess-aggregate stflife --input-csv "C:\data\ess.csv" --engine polars
```

The file must contain the requested variable and `cntry`; interview-year and
weight columns are used when available. `py-ess` reads the local file with
Polars, after which the data is converted to pandas for aggregation.
`--rounds` can be used with a local CSV if it contains the `essround` column.
Local-file mode does not download survey data from the ESS API.

### Export Parquet

Select Parquet automatically with a `.parquet` or `.pq` extension:

```powershell
ess-aggregate stflife --output "results.parquet"
```

Or explicitly select the output format:

```powershell
ess-aggregate stflife --output results.data --output-format parquet
```

## Output format

For each requested variable, the output includes four types of aggregation
units:

| `unit` | `unit_type` | Description |
|---|---|---|
| `Germany2018` | `country_year` | Germany in 2018 |
| `GermanyAll` | `country_all` | Germany, pooled across all available years |
| `All2018` | `year_all` | All countries pooled in 2018 |
| `AllAll` | `grand_all` | All countries and years pooled |

Each row in the output CSV contains:

| Column | Description |
|---|---|
| `unit` | Identifier for the aggregation unit. |
| `variable` | ESS variable name. |
| `country`, `year` | Country name and year; pooled dimensions are labelled `All`. |
| `unit_type` | `country_year`, `country_all`, `year_all`, or `grand_all`. |
| `n_raw` | Number of valid, non-missing observations before weighting. |
| `n_effective` | Kish effective sample size. |
| `reliable` | `True` when both `n_raw` and `n_effective` meet `--min-n`. |
| `mean`, `median`, `mode` | Weighted mean, median, and mode. |
| `std`, `iqr` | Weighted standard deviation and interquartile range (P75 − P25). |
| `p10`, `p40`, `p50`, `p90` | Weighted percentiles. P40 is also used to calculate the Palma ratio. |
| `p90_p10_ratio` | P90 divided by P10. |
| `palma_ratio` | Share of the total held by the top 10% divided by the share held by the bottom 40%. |
| `gini` | Weighted Gini coefficient; 0 indicates equality and 1 indicates maximum inequality. |
| `country_mean_over_years` | Mundlak between-country component, populated only for `country_year` rows. |
| `within_country_deviation` | Mundlak within-country component, populated only for `country_year` rows. |
| `warnings` | Semicolon-separated warnings, such as a low sample size. |

For Parquet output, the `year` column is stored as a string because it
contains both numeric years and the `All` pooling label.

## Calculations and business logic

### Weights

The loader prefers ESS's `anweight` (analysis weight). If it is unavailable,
it derives a combined weight from `pspwght` or `dweight` multiplied by
`pweight`, when those columns are present. If no suitable ESS weight is
available, the loader uses equal weights and logs a warning; results from that
round are therefore unweighted.

The intended role of the weights is:

- `dweight` corrects for unequal sampling probabilities.
- `pspwght` includes design and post-stratification adjustments.
- `pweight` accounts for population size and is intended to be combined with
  a design weight.
- `anweight` is the ESS-provided analysis weight.

Using a population-size component matters when pooling respondents across
countries: otherwise, countries with smaller populations can have too much
influence relative to their population. Within a single country, multiplying
all respondents' weights by the same country-level constant does not change
weighted means, percentiles, or the Kish effective sample size.

Before statistics are calculated, observations with missing/non-finite values
or non-positive/non-finite weights are excluded.

### Sample size and reliability threshold

The Kish effective sample size is calculated alongside the raw valid
observation count:

```text
n_eff = (sum(w_i))^2 / sum(w_i^2)
```

It estimates the number of equally weighted observations that would provide
the same information as the observed unequally weighted sample. It is
invariant to multiplying all weights in a group by the same constant.

If either `n_raw` or `n_effective` is below `--min-n` (default: 300),
distributional statistics are suppressed and written as `NaN`: median,
percentiles, IQR, Gini, quantile ratio, and Palma ratio. The row is retained
with `reliable = False` and a warning. Mean, mode, and standard deviation are
still reported, also with `reliable = False`.

The threshold is configurable because the appropriate minimum depends on the
analysis. A threshold between 300 and 500 may be preferable for some
applications.

### Central tendency: mean, median, and mode

- **Weighted mean:** `sum(w * x) / sum(w)`.
- **Weighted median:** weighted percentile at `q = 0.5`.
- **Weighted mode:** the observed value with the greatest total weight.
  This is most interpretable for discrete or categorical variables. For
  continuous variables it is the most heavily weighted observed value, not a
  density-estimated mode.

### Dispersion: standard deviation and IQR

- **Weighted standard deviation:** the square root of the weighted population
  variance, `sum(w * (x - weighted_mean)^2) / sum(w)`.
- **Interquartile range (IQR):** weighted P75 minus weighted P25.

Weighted percentiles use a midpoint (Hazen-style) interpolation. Values are
sorted, and each observation is assigned the midpoint of its cumulative
weight:

```text
cw_i = (cumulative_weight_i - 0.5 * w_i) / sum(w)
```

The requested quantile is linearly interpolated between the surrounding
midpoints. With equal weights, this method gives the usual midpoint-based
empirical quantile.

### Inequality: Gini, percentiles, quantile ratio, and Palma ratio

- **Weighted Gini coefficient:** calculated from the discrete Lorenz curve
  using the trapezoid formula. For values sorted in ascending order, let
  `w_share_i` be the observation's share of total weight and `L_i` the
  cumulative share of total weighted value:

  ```text
  G = 1 - sum(w_share_i * (L_(i-1) + L_i))
  ```

  The implementation is checked against a brute-force pairwise reference in
  the tests. The Gini is returned as `NaN` when any value is negative or when
  the total weighted value is zero, because this implementation assumes a
  non-negative variable.
- **P10, P50, and P90:** weighted percentiles for examining the lower tail,
  centre, and upper tail of the distribution. P40 is also reported for the
  Palma ratio.
- **P90/P10 ratio:** `P90 / P10`. It is `NaN` when P10 is zero or either
  percentile is non-finite.
- **Palma ratio:** the weighted share of the total value held by observations
  at or above P90 divided by the weighted share held by observations at or
  below P40:

  ```text
  (sum(w * x for x >= P90) / sum(w * x))
  ------------------------------------------------
  (sum(w * x for x <= P40) / sum(w * x))
  ```

  It is `NaN` when the denominator share is zero or the required values are
  undefined.

### Mundlak / zero-centering transformation

For each `country_year` unit, the output includes a within/between
decomposition of the yearly weighted mean:

- **`country_mean_over_years` (between-country component):** the arithmetic
  mean of that country's yearly aggregate means:

  ```text
  X_bar_c = (1 / T_c) * sum_t(X_ct)
  ```

  This is an unweighted average across years, not the pooled respondent-level
  mean in the `GermanyAll` row.
- **`within_country_deviation` (within-country component):** the year's
  deviation from its country's mean:

  ```text
  X_ct - X_bar_c
  ```

The within-country deviations sum to zero across a country's observed years,
up to floating-point precision. These columns are `NaN` for pooled units
because the decomposition applies to country-year observations.

### Country and year identification

- **Country:** ESS variable `cntry` contains ISO 3166-1 alpha-2 codes. The
  codebook is used to decode country names where available (for example,
  `DE` to `Germany`).
- **Year:** the loader uses the first available interview-year variable in
  this order: `inwyys` (interview start year), `inwyr` (interview year, used
  in older rounds), then `inwyye` (interview end year). Respondent-level
  interview year is used instead of a nominal round year because fieldwork
  may span calendar years.

## Error handling and logging

- Processing information and errors are written to the console and, by
  default, to `ess_aggregator.log`. The file log includes DEBUG-level detail.
- A round that cannot be downloaded or is missing a required column is logged
  and skipped; other rounds continue to be processed.
- A variable that cannot be loaded is skipped if other requested variables
  can still be processed.
- If no requested variable can be loaded, or an unexpected fatal error
  occurs, the program exits with a non-zero status and logs the error.
- Exit codes: `0` for success, `1` for a processing failure or no output rows,
  and `2` if `py-ess` is not installed.

## Tests

Run the test suite with:

```powershell
python -m pytest
```

Tests use a fake ESS client, so they do not make network requests. Coverage
includes:

- `test_aggregation.py`: weighted statistics, effective sample size,
  low-sample-size behavior, and Gini comparison with a brute-force reference.
- `test_data_loader.py`: round resolution, weight/year selection, country
  decoding, and handling of failed rounds.
- `test_pipeline.py`: aggregation units, pooling, and Mundlak decomposition.
- `test_cli.py`: command-line parsing and end-to-end output using fake data.

## Project structure

```text
ess-aggregator/
├── src/ess_aggregator/
│   ├── aggregation.py    # Weighted statistics
│   ├── cli.py            # Command-line interface
│   ├── config.py         # Weights, year columns, and thresholds
│   ├── data_loader.py    # py-ess integration and round loading
│   ├── exceptions.py     # Package-specific exceptions
│   ├── logging_utils.py  # Logging configuration
│   ├── output.py         # CSV writer
│   └── pipeline.py       # Grouping, aggregation, and Mundlak transformation
├── tests/                # pytest suite and fake ESS client
├── pyproject.toml
├── requirements.txt
└── README.md
```
