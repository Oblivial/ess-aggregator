"""Orchestration layer: turns the long respondent-level DataFrame produced by
:mod:`ess_aggregator.data_loader` into the final country-year (+ pooled)
aggregate table, including the Mundlak between/within decomposition.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

import pandas as pd

from .aggregation import aggregate_group
from .config import ALL_LABEL, DEFAULT_MIN_EFFECTIVE_N
from .exceptions import InsufficientDataError

logger = logging.getLogger("ess_aggregator")

#: Column order of the final output CSV.
OUTPUT_COLUMNS = [
    "unit",
    "variable",
    "country",
    "year",
    "unit_type",
    "n_raw",
    "n_effective",
    "reliable",
    "mean",
    "median",
    "mode",
    "std",
    "iqr",
    "p10",
    "p40",
    "p50",
    "p90",
    "p90_p10_ratio",
    "palma_ratio",
    "gini",
    "country_mean_over_years",
    "within_country_deviation",
    "warnings",
]


def _slugify(text: str) -> str:
    """Turn a country name into a safe unit-label fragment, e.g.
    ``"United Kingdom"`` -> ``"UnitedKingdom"``, so it concatenates cleanly
    with a year/``"All"`` suffix (``"UnitedKingdomAll"``, ``"UnitedKingdom2018"``)."""
    return re.sub(r"[^0-9A-Za-z]", "", text)


def _build_row(
    *,
    unit: str,
    variable: str,
    country: str,
    year: str | int,
    unit_type: str,
    values,
    weights,
    min_effective_n: int,
) -> dict:
    try:
        result = aggregate_group(values, weights, min_effective_n=min_effective_n)
    except InsufficientDataError as exc:
        logger.warning("Skipping unit=%s: %s", unit, exc)
        return {}

    row = {
        "unit": unit,
        "variable": variable,
        "country": country,
        "year": year,
        "unit_type": unit_type,
        **result.as_dict(),
        "country_mean_over_years": float("nan"),
        "within_country_deviation": float("nan"),
    }
    return row


def aggregate_variable(
    long_df: pd.DataFrame,
    variable: str,
    min_effective_n: int = DEFAULT_MIN_EFFECTIVE_N,
) -> pd.DataFrame:
    """Build every aggregation unit (country-year, country-all, year-all,
    grand-all) for a single variable's long-format data, plus the Mundlak
    between/within decomposition for the country-year rows.
    """
    df = long_df[long_df["variable"] == variable].copy()
    rows: list[dict] = []

    # -- country x year units, e.g. "Germany2018" -------------------------
    for (country, year), group in df.dropna(subset=["year"]).groupby(["country", "year"]):
        year_int = int(year)
        row = _build_row(
            unit=f"{_slugify(country)}{year_int}",
            variable=variable,
            country=country,
            year=year_int,
            unit_type="country_year",
            values=group["value"].to_numpy(),
            weights=group["weight"].to_numpy(),
            min_effective_n=min_effective_n,
        )
        if row:
            rows.append(row)

    # -- country, pooled across all years, e.g. "GermanyAll" ---------------
    for country, group in df.groupby("country"):
        row = _build_row(
            unit=f"{_slugify(country)}{ALL_LABEL}",
            variable=variable,
            country=country,
            year=ALL_LABEL,
            unit_type="country_all",
            values=group["value"].to_numpy(),
            weights=group["weight"].to_numpy(),
            min_effective_n=min_effective_n,
        )
        if row:
            rows.append(row)

    # -- all countries, pooled by year, e.g. "All2018" ----------------------
    for year, group in df.dropna(subset=["year"]).groupby("year"):
        year_int = int(year)
        row = _build_row(
            unit=f"{ALL_LABEL}{year_int}",
            variable=variable,
            country=ALL_LABEL,
            year=year_int,
            unit_type="year_all",
            values=group["value"].to_numpy(),
            weights=group["weight"].to_numpy(),
            min_effective_n=min_effective_n,
        )
        if row:
            rows.append(row)

    # -- grand total, pooled across everything, i.e. "AllAll" ---------------
    row = _build_row(
        unit=f"{ALL_LABEL}{ALL_LABEL}",
        variable=variable,
        country=ALL_LABEL,
        year=ALL_LABEL,
        unit_type="grand_all",
        values=df["value"].to_numpy(),
        weights=df["weight"].to_numpy(),
        min_effective_n=min_effective_n,
    )
    if row:
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return _apply_mundlak(result)


def _apply_mundlak(result: pd.DataFrame) -> pd.DataFrame:
    """Add the Mundlak between/within decomposition to country-year rows.

    Following Mundlak (1978), the macro-predictor's *between*-country
    component is the simple average of the country's own yearly statistic
    across all years it has data for (``country_mean_over_years``), and the
    *within*-country component is each year's deviation from that country
    mean (``within_country_deviation``). Both are computed on the weighted
    mean (the standard central-tendency macro-indicator); see README.md for
    the full rationale. Only ``country_year`` rows get non-NaN values here -
    the concept doesn't apply to the pooled (``*_all``) units.
    """
    is_country_year = result["unit_type"] == "country_year"
    country_means = result.loc[is_country_year].groupby("country")["mean"].transform("mean")
    result.loc[is_country_year, "country_mean_over_years"] = country_means
    result.loc[is_country_year, "within_country_deviation"] = (
        result.loc[is_country_year, "mean"] - country_means
    )
    return result


def aggregate_variables(
    long_df: pd.DataFrame,
    variables: Iterable[str],
    min_effective_n: int = DEFAULT_MIN_EFFECTIVE_N,
) -> pd.DataFrame:
    """Aggregate every requested variable and concatenate the results."""
    frames = [
        aggregate_variable(long_df, variable, min_effective_n=min_effective_n)
        for variable in variables
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined[OUTPUT_COLUMNS]


def run_pipeline(
    ess_client,
    variables: list[str],
    requested_rounds: list[str] | None = None,
    countries: list[str] | None = None,
    min_effective_n: int = DEFAULT_MIN_EFFECTIVE_N,
    recode_missing_values: bool = True,
    local_csv_path: str | None = None,
    engine: str = "pandas",
) -> pd.DataFrame:
    """End-to-end pipeline: load every requested variable (across every
    applicable round) via ``py-ess``, then aggregate.

    Loading errors for individual variables are logged and that variable is
    skipped (partial-success error handling) rather than aborting the run;
    a variable that fails for *every* round raises inside the loader and is
    caught here so the remaining variables can still be processed.
    """
    from .data_loader import EssDataLoader
    from .exceptions import ESSAggregatorError

    loader = EssDataLoader(
        ess_client,
        recode_missing_values=recode_missing_values,
        local_csv_path=local_csv_path,
        engine=engine,
    )
    long_frames = []
    for variable in variables:
        try:
            long_frames.append(
                loader.load_variable_long(variable, requested_rounds, countries)
            )
        except ESSAggregatorError as exc:
            logger.error("Skipping variable %r entirely: %s", variable, exc)

    if not long_frames:
        raise ESSAggregatorError("No data could be loaded for any requested variable.")

    long_df = pd.concat(long_frames, ignore_index=True)
    return aggregate_variables(long_df, variables, min_effective_n=min_effective_n)
