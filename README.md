# ess-aggregator

Turn European Social Survey (ESS) respondent data into country-year aggregates
for multilevel and panel analyses. Results can be saved as CSV or Parquet.

## Quick start

Install the package:

```powershell
python -m pip install .
```

Get an ESS API user ID at [ess.sikt.no/en/api](https://ess.sikt.no/en/api),
then set it for the current PowerShell session and run an aggregation:

```powershell
$env:PYESS_USER_ID = "your-ess-user-id"
ess-aggregate stfeco --output results.csv
```

The command downloads the ESS rounds containing `stfeco` and writes the
aggregates to `results.csv`. The `--output` option is optional; by default,
results are written to `ess_aggregated.csv`.

## Common options

```powershell
# Aggregate multiple variables, restrict rounds or countries
ess-aggregate stfeco stflife --rounds ESS9 ESS10 --countries DE FR

# Load a local ESS CSV using Polars (optional dependency)
python -m pip install ".[polars]"
ess-aggregate stfeco --input-csv "C:\data\ess.csv" --engine polars

# Write Parquet (also selected automatically by a .parquet or .pq extension)
ess-aggregate stfeco --output results.parquet
```

Use `ess-aggregate --help` to see all options. Local CSV files must include
the requested variable and `cntry`; interview-year and weight columns are
used when present. Local input does not access the ESS API. To filter a local
file by round, it must also include `essround`.

## Output at a glance

Each variable produces country-year rows and pooled rows:

| Example unit | Meaning |
|---|---|
| `Germany2018` | Germany in 2018 |
| `GermanyAll` | Germany across all available years |
| `All2018` | All countries in 2018 |
| `AllAll` | All countries and years |

The output includes weighted mean, median, mode, standard deviation, IQR,
P10/P40/P50/P90, P90/P10 ratio, Palma ratio, and Gini, along with raw and
effective sample sizes. It also includes the country mean across years and
the within-country deviation for country-year rows.

## Calculation notes

- **Weights:** `anweight` is preferred. If unavailable, a combined
  design/population weight is derived when possible; otherwise the run logs a
  warning and uses equal weights. Missing values and invalid weights are
  excluded.
- **Small samples:** the default minimum is 300. If either the raw or Kish
  effective sample size is below `--min-n`, distributional statistics
  (including quantiles, Gini, IQR, and ratios) are reported as `NaN` and the
  row is marked `reliable = False`. Mean, mode, and standard deviation remain
  available.
- **Effective sample size:** `n_eff = (sum(w))² / sum(w²)`.
- **Mundlak decomposition:** for each country, the between component is the
  simple average of its yearly means; the within component is each yearly mean
  minus that country average. Pooled rows do not have these components.
- **Interview year:** the loader uses the first available of `inwyys`,
  `inwyr`, and `inwyye`. Country codes in `cntry` are decoded using the ESS
  codebook when possible.

The default log file is `ess_aggregator.log`. Failed rounds are logged and
skipped; the run exits with an error if no requested variable can be loaded.

<details>
<summary>Development and tests</summary>

Create a virtual environment and install the package with development
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

Tests use a fake ESS client and do not make API requests. When developing
`py-ess` and `ess-aggregator` together, use `scripts/dev-setup.ps1` to install
both local checkouts in editable mode.

</details>
